-- =============================================================================
-- KrafGPU-X1 Branch Unit + SIMT Divergence Stack
-- Project      : KrafGPU-X1 Next-Generation GPU Architecture
-- Author       : Krafvyn Tech / Don
-- Description  : Handles all control flow for SIMT execution.
--
--   Branch Unit:
--     - Evaluates predicate registers to determine branch direction
--     - Calculates branch target PC = PC + (imm << 2)
--     - Issues pipeline flush + new PC when taken
--     - Detects convergence points (SYNC/BAR instructions)
--
--   SIMT Divergence Stack:
--     - Maintains per-warp active thread mask (32-bit for WARP_SIZE=32)
--     - On divergence (threads disagree on branch): pushes reconvergence PC
--       and inactive mask onto the stack
--     - On convergence (SYNC/BAR): pops stack and restores full mask
--     - Depth: 8 levels (handles nested if/else, loops)
--
--   This is the mechanism that lets 32 threads in a warp take different
--   branches while hardware serialises the two paths correctly.
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_branch_unit is
  generic (
    SM_ID       : integer := 0;
    NUM_WARPS   : integer := MAX_WARPS_PER_SM   -- 64
  );
  port (
    clk             : in  std_logic;
    clk_en          : in  std_logic;
    rst_n           : in  std_logic;

    -- Instruction from decode stage
    valid_in        : in  std_logic;
    warp_id_in      : in  unsigned(5 downto 0);
    opcode_in       : in  opcode_t;
    pc_in           : in  unsigned(PC_WIDTH-1 downto 0);
    imm_in          : in  signed(15 downto 0);

    -- Predicate register value for this warp
    pred_val_in     : in  std_logic;              -- Evaluated predicate

    -- Thread vote: which of the 32 threads agree with predicate
    thread_vote_in  : in  std_logic_vector(WARP_SIZE-1 downto 0);

    -- Pipeline control outputs
    branch_taken    : out std_logic;
    branch_target   : out unsigned(PC_WIDTH-1 downto 0);
    flush_out       : out std_logic;

    -- Active thread mask output (used by execution units to lane-gate)
    active_mask_out : out std_logic_vector(WARP_SIZE-1 downto 0);

    -- Warp state update to scheduler
    warp_sync_done  : out std_logic;
    warp_id_sync    : out unsigned(5 downto 0)
  );
end entity krafgpu_branch_unit;

architecture rtl of krafgpu_branch_unit is

  -- ===========================================================================
  -- DIVERGENCE STACK
  -- Per warp: 8-level stack of (reconvergence_pc, active_mask)
  -- ===========================================================================
  constant STACK_DEPTH  : integer := 8;

  type stack_entry_t is record
    reconv_pc   : unsigned(PC_WIDTH-1 downto 0);
    saved_mask  : std_logic_vector(WARP_SIZE-1 downto 0);
    valid       : std_logic;
  end record;

  type stack_t      is array (0 to STACK_DEPTH-1) of stack_entry_t;
  type warp_stack_t is array (0 to NUM_WARPS-1)   of stack_t;
  type warp_mask_t  is array (0 to NUM_WARPS-1)   of
                       std_logic_vector(WARP_SIZE-1 downto 0);
  type stack_ptr_t  is array (0 to NUM_WARPS-1)   of
                       integer range 0 to STACK_DEPTH-1;

  signal div_stack    : warp_stack_t  := (others => (others => (
                          reconv_pc  => (others => '0'),
                          saved_mask => (others => '1'),
                          valid      => '0')));
  signal active_mask  : warp_mask_t  := (others => (others => '1'));
  signal stack_ptr    : stack_ptr_t  := (others => 0);

  -- ===========================================================================
  -- INTERNAL SIGNALS
  -- ===========================================================================
  signal all_taken    : std_logic;  -- All 32 threads take branch
  signal all_not_taken: std_logic;  -- All 32 threads fall through
  signal diverged     : std_logic;  -- Threads disagree

  signal branch_taken_r : std_logic := '0';
  signal branch_target_r: unsigned(PC_WIDTH-1 downto 0) := (others => '0');
  signal flush_r        : std_logic := '0';

begin

  -- ===========================================================================
  -- VOTE ANALYSIS
  -- ===========================================================================
  all_taken     <= '1' when thread_vote_in = (WARP_SIZE-1 downto 0 => '1')
                       else '0';
  all_not_taken <= '1' when thread_vote_in = (WARP_SIZE-1 downto 0 => '0')
                       else '0';
  diverged      <= '1' when all_taken = '0' and all_not_taken = '0'
                       else '0';

  -- ===========================================================================
  -- BRANCH + DIVERGENCE STATE MACHINE
  -- ===========================================================================
  p_branch : process(clk)
    variable wid        : integer range 0 to NUM_WARPS-1;
    variable sp         : integer range 0 to STACK_DEPTH-1;
    variable taken_mask : std_logic_vector(WARP_SIZE-1 downto 0);
    variable fall_mask  : std_logic_vector(WARP_SIZE-1 downto 0);
    variable target_pc  : unsigned(PC_WIDTH-1 downto 0);
  begin
    if rising_edge(clk) and clk_en = '1' then
      if rst_n = '0' then
        branch_taken_r  <= '0';
        branch_target_r <= (others => '0');
        flush_r         <= '0';
        warp_sync_done  <= '0';
        warp_id_sync    <= (others => '0');
        for w in 0 to NUM_WARPS-1 loop
          active_mask(w) <= (others => '1');
          stack_ptr(w)   <= 0;
        end loop;

      else
        branch_taken_r <= '0';
        flush_r        <= '0';
        warp_sync_done <= '0';

        if valid_in = '1' then
          wid       := to_integer(warp_id_in);
          sp        := stack_ptr(wid);
          target_pc := pc_in + unsigned(resize(imm_in & "00", PC_WIDTH));

          case opcode_in is

            -- ----------------------------------------------------------------
            -- BRA (conditional branch)
            -- ----------------------------------------------------------------
            when OP_BRA =>

              if all_taken = '1' then
                -- Uniform branch: all threads agree → simple taken branch
                branch_taken_r  <= '1';
                branch_target_r <= target_pc;
                flush_r         <= '1';

              elsif all_not_taken = '1' then
                -- Uniform fall-through: no action, PC already incremented
                branch_taken_r  <= '0';

              else
                -- DIVERGENCE: push reconvergence PC onto stack
                -- Reconvergence PC = fall-through PC (pc_in + 4, already in pipeline)
                if sp < STACK_DEPTH - 1 then
                  -- Save current mask and fall-through threads
                  div_stack(wid)(sp).reconv_pc  <= pc_in + 4;
                  div_stack(wid)(sp).saved_mask  <=
                      not thread_vote_in and active_mask(wid);
                  div_stack(wid)(sp).valid       <= '1';
                  stack_ptr(wid)                 <= sp + 1;
                end if;

                -- Execute taken-branch threads first
                taken_mask       := thread_vote_in and active_mask(wid);
                active_mask(wid) <= taken_mask;
                branch_taken_r   <= '1';
                branch_target_r  <= target_pc;
                flush_r          <= '1';
              end if;

            -- ----------------------------------------------------------------
            -- SYNC: Warp-level barrier — threads reconverge here
            -- ----------------------------------------------------------------
            when OP_SYNC | OP_BAR =>

              if sp > 0 and div_stack(wid)(sp-1).valid = '1' then
                -- Pop stack: restore the other half of threads
                active_mask(wid) <= div_stack(wid)(sp-1).saved_mask;
                stack_ptr(wid)   <= sp - 1;
                -- Branch to reconvergence PC
                branch_taken_r   <= '1';
                branch_target_r  <= div_stack(wid)(sp-1).reconv_pc;
                flush_r          <= '1';
                div_stack(wid)(sp-1).valid <= '0';
              else
                -- Stack empty: all threads converged, resume full mask
                active_mask(wid) <= (others => '1');
                warp_sync_done   <= '1';
                warp_id_sync     <= warp_id_in;
              end if;

            -- ----------------------------------------------------------------
            -- CALL: Push return address, branch to target
            -- ----------------------------------------------------------------
            when OP_CALL =>
              if sp < STACK_DEPTH - 1 then
                div_stack(wid)(sp).reconv_pc  <= pc_in + 4;  -- Return addr
                div_stack(wid)(sp).saved_mask  <= active_mask(wid);
                div_stack(wid)(sp).valid       <= '1';
                stack_ptr(wid)                 <= sp + 1;
              end if;
              branch_taken_r  <= '1';
              branch_target_r <= target_pc;
              flush_r         <= '1';

            -- ----------------------------------------------------------------
            -- RET: Pop return address from stack
            -- ----------------------------------------------------------------
            when OP_RET =>
              if sp > 0 then
                branch_taken_r  <= '1';
                branch_target_r <= div_stack(wid)(sp-1).reconv_pc;
                flush_r         <= '1';
                active_mask(wid)<= div_stack(wid)(sp-1).saved_mask;
                stack_ptr(wid)  <= sp - 1;
                div_stack(wid)(sp-1).valid <= '0';
              end if;

            when others =>
              null;
          end case;
        end if;
      end if;
    end if;
  end process p_branch;

  -- ===========================================================================
  -- OUTPUTS
  -- ===========================================================================
  branch_taken    <= branch_taken_r;
  branch_target   <= branch_target_r;
  flush_out       <= flush_r;

  -- Active mask for the currently executing warp
  active_mask_out <= active_mask(to_integer(warp_id_in))
                     when valid_in = '1'
                     else (others => '1');

end architecture rtl;
