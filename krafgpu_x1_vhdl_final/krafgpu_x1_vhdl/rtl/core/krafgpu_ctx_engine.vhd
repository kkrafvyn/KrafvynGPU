-- =============================================================================
-- KrafGPU-X1 Context Save / Restore Engine
-- Project      : KrafGPU-X1 Next-Generation GPU Architecture
-- Author       : Krafvyn Tech / Don
-- Description  : GPU context save and restore for preemptive multitasking.
--
--   Context state includes (per SM × 512):
--     - Warp PC values            (64 warps × 32-bit = 256 B per SM)
--     - Active warp valid mask    (64-bit)
--     - Warp state codes          (64 × 3-bit)
--     - Divergence stack          (8 levels × 64-bit per warp)
--     - Active thread masks       (64 warps × 32-bit)
--     - Predicate registers       (64 warps × 8-bit)
--     - Texture unit state        (current LOD, sampler bindings)
--
--   Register file (64 KB per SM) is already in SRAM and is preserved
--   in-place if SM power is maintained. Only saved if SM is powered off.
--
--   Save procedure:
--     1. Drain pipeline (wait for all in-flight warps to complete/stall)
--     2. Snapshot warp PCs and states via DMA to context buffer in VRAM
--     3. Optionally dump register file (if power-gating SM)
--     4. Signal driver: context save complete (fence)
--
--   Restore procedure:
--     1. DMA load context buffer from VRAM → SM state registers
--     2. Reload register file if it was saved
--     3. Resume warp scheduler
--     4. Signal driver: context restore complete (fence)
--
--   Context buffer layout in VRAM (per SM, 4 KB):
--     0x000  Warp PCs     (64 × 4B = 256B)
--     0x100  Warp states  (64 × 4B = 256B)
--     0x200  Thread masks (64 × 4B = 256B)
--     0x300  Pred regs    (64 × 4B = 256B)  [1B padded to 4B]
--     0x400  Div stack    (64 × 8 × 8B = 4KB) [64warps × 8levels × 8B]
--     Total  ~8 KB per SM → 512 SMs × 8 KB = 4 MB per full context
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_ctx_engine is
  generic (
    NUM_SMS_LOCAL  : integer := 512;
    CTX_BYTES_PER_SM : integer := 8192   -- 8 KB per SM
  );
  port (
    clk              : in  std_logic;
    rst_n            : in  std_logic;

    -- Control from driver (via BAR0 / command ring)
    ctx_save_req     : in  std_logic;
    ctx_restore_req  : in  std_logic;
    ctx_buf_addr     : in  unsigned(PHYS_ADDR_WIDTH-1 downto 0); -- Base in VRAM
    ctx_sm_mask      : in  std_logic_vector(NUM_ISLANDS-1 downto 0); -- Which SMs

    -- Pipeline drain control
    drain_req        : out std_logic;
    drain_done       : in  std_logic;

    -- SM state snapshot interface (read warp PCs etc.)
    sm_select        : out integer range 0 to NUM_SMS_LOCAL-1;
    warp_pc_in       : in  unsigned(PC_WIDTH-1 downto 0);
    warp_state_in    : in  warp_state_t;
    warp_mask_in     : in  std_logic_vector(WARP_SIZE-1 downto 0);
    pred_reg_in      : in  std_logic_vector(7 downto 0);

    -- SM state restore interface (write warp PCs etc.)
    warp_pc_out      : out unsigned(PC_WIDTH-1 downto 0);
    warp_state_out   : out warp_state_t;
    warp_restore_en  : out std_logic;
    warp_restore_id  : out unsigned(5 downto 0);
    sm_restore_id    : out integer range 0 to NUM_SMS_LOCAL-1;

    -- DMA interface (to/from VRAM context buffer)
    dma_req          : out mem_req_t;
    dma_resp         : in  mem_resp_t;

    -- Status / completion
    ctx_save_done    : out std_logic;
    ctx_restore_done : out std_logic;
    ctx_busy         : out std_logic;

    -- Register file save (separate DMA, larger transfer)
    regfile_save_req  : out std_logic;
    regfile_save_addr : out unsigned(PHYS_ADDR_WIDTH-1 downto 0);
    regfile_save_sm   : out integer range 0 to NUM_SMS_LOCAL-1;
    regfile_save_done : in  std_logic
  );
end entity krafgpu_ctx_engine;

architecture rtl of krafgpu_ctx_engine is

  -- ===========================================================================
  -- STATE MACHINE
  -- ===========================================================================
  type ctx_state_t is (
    CTX_IDLE,
    -- Save sequence
    CTX_DRAIN,          -- Wait for pipeline to drain
    CTX_SAVE_PC,        -- DMA warp PC for current warp
    CTX_SAVE_STATE,     -- DMA warp state
    CTX_SAVE_MASK,      -- DMA active thread mask
    CTX_SAVE_PRED,      -- DMA predicate registers
    CTX_SAVE_NEXT_WARP, -- Advance to next warp
    CTX_SAVE_NEXT_SM,   -- Advance to next SM
    CTX_SAVE_REGFILE,   -- Save register file (if needed)
    CTX_SAVE_FENCE,     -- Write completion fence
    CTX_SAVE_FSM_DONE,  -- Signal completion
    -- Restore sequence
    CTX_RESTORE_INIT,
    CTX_RESTORE_PC,
    CTX_RESTORE_STATE,
    CTX_RESTORE_MASK,
    CTX_RESTORE_PRED,
    CTX_RESTORE_APPLY,
    CTX_RESTORE_NEXT_WARP,
    CTX_RESTORE_NEXT_SM,
    CTX_RESTORE_REGFILE,
    CTX_RESTORE_FSM_DONE
  );
  signal ctx_state  : ctx_state_t := CTX_IDLE;

  -- ===========================================================================
  -- LOOP COUNTERS
  -- ===========================================================================
  signal cur_sm     : integer range 0 to NUM_SMS_LOCAL-1 := 0;
  signal cur_warp   : integer range 0 to MAX_WARPS_PER_SM-1 := 0;

  -- ===========================================================================
  -- STAGING REGISTERS (hold values between DMA and apply)
  -- ===========================================================================
  signal saved_pc   : unsigned(PC_WIDTH-1 downto 0);
  signal saved_state: warp_state_t;
  signal saved_mask : std_logic_vector(WARP_SIZE-1 downto 0);
  signal saved_pred : std_logic_vector(7 downto 0);

  -- ===========================================================================
  -- DMA ADDRESS HELPERS
  -- Compute VRAM address for each field based on SM and warp index
  -- ===========================================================================
  function ctx_pc_addr(base    : unsigned(PHYS_ADDR_WIDTH-1 downto 0);
                       sm_id   : integer;
                       warp_id : integer)
    return unsigned is
  begin
    return base +
      to_unsigned(sm_id * CTX_BYTES_PER_SM, PHYS_ADDR_WIDTH) +
      to_unsigned(warp_id * 4, PHYS_ADDR_WIDTH);
  end function;

  function ctx_state_addr(base    : unsigned(PHYS_ADDR_WIDTH-1 downto 0);
                          sm_id   : integer;
                          warp_id : integer)
    return unsigned is
  begin
    return base +
      to_unsigned(sm_id * CTX_BYTES_PER_SM + 256, PHYS_ADDR_WIDTH) +
      to_unsigned(warp_id * 4, PHYS_ADDR_WIDTH);
  end function;

  function ctx_mask_addr(base    : unsigned(PHYS_ADDR_WIDTH-1 downto 0);
                         sm_id   : integer;
                         warp_id : integer)
    return unsigned is
  begin
    return base +
      to_unsigned(sm_id * CTX_BYTES_PER_SM + 512, PHYS_ADDR_WIDTH) +
      to_unsigned(warp_id * 4, PHYS_ADDR_WIDTH);
  end function;

  function ctx_pred_addr(base    : unsigned(PHYS_ADDR_WIDTH-1 downto 0);
                         sm_id   : integer;
                         warp_id : integer)
    return unsigned is
  begin
    return base +
      to_unsigned(sm_id * CTX_BYTES_PER_SM + 768, PHYS_ADDR_WIDTH) +
      to_unsigned(warp_id * 4, PHYS_ADDR_WIDTH);
  end function;

  -- Encode warp_state_t as 4-bit integer for storage
  function state_to_slv(s : warp_state_t) return std_logic_vector is
  begin
    case s is
      when WARP_IDLE        => return x"00000000";
      when WARP_READY       => return x"00000001";
      when WARP_RUNNING     => return x"00000002";
      when WARP_STALLED_MEM => return x"00000003";
      when WARP_STALLED_DEP => return x"00000004";
      when WARP_BARRIER     => return x"00000005";
      when WARP_DONE        => return x"00000006";
    end case;
  end function;

  function slv_to_state(v : std_logic_vector(31 downto 0)) return warp_state_t is
  begin
    case v(2 downto 0) is
      when "000"  => return WARP_IDLE;
      when "001"  => return WARP_READY;
      when "010"  => return WARP_RUNNING;
      when "011"  => return WARP_STALLED_MEM;
      when "100"  => return WARP_STALLED_DEP;
      when "101"  => return WARP_BARRIER;
      when others => return WARP_DONE;
    end case;
  end function;

  -- Output registers
  signal save_done_r    : std_logic := '0';
  signal restore_done_r : std_logic := '0';
  signal busy_r         : std_logic := '0';
  signal drain_req_r    : std_logic := '0';
  signal restore_en_r   : std_logic := '0';

begin

  sm_select     <= cur_sm;
  ctx_save_done    <= save_done_r;
  ctx_restore_done <= restore_done_r;
  ctx_busy         <= busy_r;
  drain_req        <= drain_req_r;
  warp_restore_en  <= restore_en_r;
  warp_restore_id  <= to_unsigned(cur_warp, 6);
  sm_restore_id    <= cur_sm;
  warp_pc_out      <= saved_pc;
  warp_state_out   <= saved_state;

  -- ===========================================================================
  -- CONTEXT ENGINE STATE MACHINE
  -- ===========================================================================
  p_ctx_fsm : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        ctx_state      <= CTX_IDLE;
        cur_sm         <= 0;
        cur_warp       <= 0;
        save_done_r    <= '0';
        restore_done_r <= '0';
        busy_r         <= '0';
        drain_req_r    <= '0';
        restore_en_r   <= '0';
        dma_req.valid  <= '0';
        regfile_save_req <= '0';
      else
        save_done_r    <= '0';
        restore_done_r <= '0';
        dma_req.valid  <= '0';
        restore_en_r   <= '0';
        regfile_save_req <= '0';

        case ctx_state is

          -- ----------------------------------------------------------------
          when CTX_IDLE =>
            busy_r <= '0';
            if ctx_save_req = '1' then
              cur_sm     <= 0;
              cur_warp   <= 0;
              busy_r     <= '1';
              ctx_state  <= CTX_DRAIN;
            elsif ctx_restore_req = '1' then
              cur_sm     <= 0;
              cur_warp   <= 0;
              busy_r     <= '1';
              ctx_state  <= CTX_RESTORE_INIT;
            end if;

          -- ----------------------------------------------------------------
          -- SAVE: Step 1 — drain pipeline
          -- ----------------------------------------------------------------
          when CTX_DRAIN =>
            drain_req_r <= '1';
            if drain_done = '1' then
              drain_req_r <= '0';
              ctx_state   <= CTX_SAVE_PC;
            end if;

          -- ----------------------------------------------------------------
          -- SAVE: Step 2 — DMA warp PC
          -- ----------------------------------------------------------------
          when CTX_SAVE_PC =>
            dma_req.valid    <= '1';
            dma_req.addr     <= ctx_pc_addr(ctx_buf_addr, cur_sm, cur_warp);
            dma_req.write_en <= '1';
            dma_req.data     <= std_logic_vector(warp_pc_in);
            dma_req.byte_en  <= "1111";
            dma_req.warp_id  <= (others => '0');
            dma_req.req_id   <= to_unsigned(0, 8);
            ctx_state        <= CTX_SAVE_STATE;

          when CTX_SAVE_STATE =>
            if dma_resp.valid = '1' or dma_req.valid = '0' then
              dma_req.valid    <= '1';
              dma_req.addr     <= ctx_state_addr(ctx_buf_addr, cur_sm, cur_warp);
              dma_req.write_en <= '1';
              dma_req.data     <= state_to_slv(warp_state_in);
              dma_req.byte_en  <= "1111";
              dma_req.warp_id  <= (others => '0');
              dma_req.req_id   <= to_unsigned(1, 8);
              ctx_state        <= CTX_SAVE_MASK;
            end if;

          when CTX_SAVE_MASK =>
            if dma_resp.valid = '1' or dma_req.valid = '0' then
              dma_req.valid    <= '1';
              dma_req.addr     <= ctx_mask_addr(ctx_buf_addr, cur_sm, cur_warp);
              dma_req.write_en <= '1';
              dma_req.data     <= warp_mask_in;
              dma_req.byte_en  <= "1111";
              dma_req.warp_id  <= (others => '0');
              dma_req.req_id   <= to_unsigned(2, 8);
              ctx_state        <= CTX_SAVE_PRED;
            end if;

          when CTX_SAVE_PRED =>
            if dma_resp.valid = '1' or dma_req.valid = '0' then
              dma_req.valid    <= '1';
              dma_req.addr     <= ctx_pred_addr(ctx_buf_addr, cur_sm, cur_warp);
              dma_req.write_en <= '1';
              dma_req.data     <= (31 downto 8 => '0') & pred_reg_in;
              dma_req.byte_en  <= "0001";
              dma_req.warp_id  <= (others => '0');
              dma_req.req_id   <= to_unsigned(3, 8);
              ctx_state        <= CTX_SAVE_NEXT_WARP;
            end if;

          when CTX_SAVE_NEXT_WARP =>
            if dma_resp.valid = '1' or dma_req.valid = '0' then
              if cur_warp < MAX_WARPS_PER_SM - 1 then
                cur_warp  <= cur_warp + 1;
                ctx_state <= CTX_SAVE_PC;
              else
                cur_warp  <= 0;
                ctx_state <= CTX_SAVE_NEXT_SM;
              end if;
            end if;

          when CTX_SAVE_NEXT_SM =>
            if cur_sm < NUM_SMS_LOCAL - 1 then
              cur_sm    <= cur_sm + 1;
              ctx_state <= CTX_SAVE_PC;
            else
              cur_sm    <= 0;
              ctx_state <= CTX_SAVE_REGFILE;
            end if;

          when CTX_SAVE_REGFILE =>
            -- Trigger register file DMA save for each SM
            regfile_save_req  <= '1';
            regfile_save_sm   <= cur_sm;
            regfile_save_addr <= ctx_buf_addr +
              to_unsigned(cur_sm * CTX_BYTES_PER_SM + 4096,
                          PHYS_ADDR_WIDTH);
            if regfile_save_done = '1' then
              if cur_sm < NUM_SMS_LOCAL - 1 then
                cur_sm    <= cur_sm + 1;
              else
                ctx_state <= CTX_SAVE_FENCE;
              end if;
            end if;

          when CTX_SAVE_FENCE =>
            -- Write save-complete magic to fence word in context buffer
            dma_req.valid    <= '1';
            dma_req.addr     <= ctx_buf_addr + 4;  -- Fence DWORD at offset 4
            dma_req.write_en <= '1';
            dma_req.data     <= x"C0FFEE01";  -- Magic: context saved v1
            dma_req.byte_en  <= "1111";
            dma_req.warp_id  <= (others => '0');
            dma_req.req_id   <= to_unsigned(255, 8);
            ctx_state        <= CTX_SAVE_FSM_DONE;

          when CTX_SAVE_FSM_DONE =>
            if dma_resp.valid = '1' then
              save_done_r <= '1';
              busy_r      <= '0';
              ctx_state   <= CTX_IDLE;
            end if;

          -- ----------------------------------------------------------------
          -- RESTORE SEQUENCE
          -- ----------------------------------------------------------------
          when CTX_RESTORE_INIT =>
            cur_sm    <= 0;
            cur_warp  <= 0;
            ctx_state <= CTX_RESTORE_PC;

          when CTX_RESTORE_PC =>
            dma_req.valid    <= '1';
            dma_req.addr     <= ctx_pc_addr(ctx_buf_addr, cur_sm, cur_warp);
            dma_req.write_en <= '0';
            dma_req.data     <= (others => '0');
            dma_req.byte_en  <= "1111";
            dma_req.warp_id  <= (others => '0');
            dma_req.req_id   <= to_unsigned(0, 8);
            ctx_state        <= CTX_RESTORE_STATE;

          when CTX_RESTORE_STATE =>
            if dma_resp.valid = '1' then
              saved_pc  <= unsigned(dma_resp.data);
              dma_req.valid    <= '1';
              dma_req.addr     <= ctx_state_addr(ctx_buf_addr, cur_sm, cur_warp);
              dma_req.write_en <= '0';
              dma_req.data     <= (others => '0');
              dma_req.byte_en  <= "1111";
              dma_req.warp_id  <= (others => '0');
              dma_req.req_id   <= to_unsigned(1, 8);
              ctx_state        <= CTX_RESTORE_MASK;
            end if;

          when CTX_RESTORE_MASK =>
            if dma_resp.valid = '1' then
              saved_state <= slv_to_state(dma_resp.data);
              dma_req.valid    <= '1';
              dma_req.addr     <= ctx_mask_addr(ctx_buf_addr, cur_sm, cur_warp);
              dma_req.write_en <= '0';
              dma_req.data     <= (others => '0');
              dma_req.byte_en  <= "1111";
              dma_req.warp_id  <= (others => '0');
              dma_req.req_id   <= to_unsigned(2, 8);
              ctx_state        <= CTX_RESTORE_PRED;
            end if;

          when CTX_RESTORE_PRED =>
            if dma_resp.valid = '1' then
              saved_mask <= dma_resp.data;
              dma_req.valid    <= '1';
              dma_req.addr     <= ctx_pred_addr(ctx_buf_addr, cur_sm, cur_warp);
              dma_req.write_en <= '0';
              dma_req.data     <= (others => '0');
              dma_req.byte_en  <= "0001";
              dma_req.warp_id  <= (others => '0');
              dma_req.req_id   <= to_unsigned(3, 8);
              ctx_state        <= CTX_RESTORE_APPLY;
            end if;

          when CTX_RESTORE_APPLY =>
            if dma_resp.valid = '1' then
              saved_pred   <= dma_resp.data(7 downto 0);
              restore_en_r <= '1';   -- Signal SM to load saved_pc/state/mask
              ctx_state    <= CTX_RESTORE_NEXT_WARP;
            end if;

          when CTX_RESTORE_NEXT_WARP =>
            if cur_warp < MAX_WARPS_PER_SM - 1 then
              cur_warp  <= cur_warp + 1;
              ctx_state <= CTX_RESTORE_PC;
            else
              cur_warp  <= 0;
              ctx_state <= CTX_RESTORE_NEXT_SM;
            end if;

          when CTX_RESTORE_NEXT_SM =>
            if cur_sm < NUM_SMS_LOCAL - 1 then
              cur_sm    <= cur_sm + 1;
              ctx_state <= CTX_RESTORE_PC;
            else
              cur_sm    <= 0;
              ctx_state <= CTX_RESTORE_REGFILE;
            end if;

          when CTX_RESTORE_REGFILE =>
            -- Restore register files (large DMA)
            regfile_save_req  <= '1';  -- Reusing same signal (direction set by state)
            regfile_save_sm   <= cur_sm;
            regfile_save_addr <= ctx_buf_addr +
              to_unsigned(cur_sm * CTX_BYTES_PER_SM + 4096,
                          PHYS_ADDR_WIDTH);
            if regfile_save_done = '1' then
              if cur_sm < NUM_SMS_LOCAL - 1 then
                cur_sm    <= cur_sm + 1;
              else
                ctx_state <= CTX_RESTORE_FSM_DONE;
              end if;
            end if;

          when CTX_RESTORE_FSM_DONE =>
            restore_done_r <= '1';
            busy_r         <= '0';
            ctx_state      <= CTX_IDLE;

        end case;
      end if;
    end if;
  end process p_ctx_fsm;

end architecture rtl;
