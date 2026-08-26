-- =============================================================================
-- KrafGPU-X1 Pixel-Quad Warp Dispatcher
-- Project      : KrafGPU-X1 Next-Generation GPU Architecture
-- Author       : Krafvyn Tech / Don
-- Description  : Bridges the rasterisation pipeline's per-pixel output to
--                the SM pixel-shader dispatch interface.
--
--   The fine rasteriser + depth test produces one pixel per clock. This
--   module collects 4 consecutive surviving pixels into a 2×2 quad (the
--   minimum unit of pixel-shader execution for SIMT coercion and ddx/ddy
--   derivative computation), then fires a ps_valid pulse to the target SM
--   with the quad's position and barycentric attributes.
--
--   Warp-to-SM routing:
--     The dispatcher maintains a round-robin SM selection counter
--     (sm_rr_ptr, 0..NUM_SMS_TOTAL-1). On each quad dispatch, it targets
--     the next SM in round-robin order, favouring SMs that have reported
--     ps_ready = '1'. The selected SM receives:
--       - ps_warp_id  : a warp ID within that SM (round-robin 0..3,
--                       giving at most 4 concurrent pixel warps per SM)
--       - ps_screen_x/y: top-left pixel of the 2×2 quad
--       - ps_bary_u/v  : barycentric coordinates of pixel (0,0) in the quad
--       - ps_depth     : interpolated depth of pixel (0,0)
--
--   In a production GPU, additional "helper threads" for pixels at quad
--   boundaries that fail coverage tests are launched but flagged as
--   "discarded" so ddx/ddy are still valid; that complexity is omitted
--   here — all 4 pixels in a quad are assumed to pass the depth test.
--
--   Current limitation (documented):
--     The SM ps_valid/ps_ready interface added in krafgpu_sm v2 only
--     exposes a single warp-dispatch port. A full implementation would
--     use a dispatch queue per SM, allowing the rasteriser to outrun the
--     SM's per-cycle acceptance rate. This is left as a follow-on extension.
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_ps_dispatch is
  generic (
    NUM_SMS         : integer := NUM_SMS_TOTAL;   -- Total SMs across both chiplets
    WARPS_PER_SM    : integer := 4                -- Max concurrent pixel-shader warps per SM
  );
  port (
    clk             : in  std_logic;
    rst_n           : in  std_logic;

    -- From depth-buffer (surviving pixels, one per cycle max)
    pix_valid       : in  std_logic;
    pix_x           : in  unsigned(15 downto 0);
    pix_y           : in  unsigned(15 downto 0);
    pix_cov         : in  std_logic_vector(3 downto 0);  -- MSAA sample mask
    pix_bary_u      : in  std_logic_vector(DATA_WIDTH-1 downto 0);
    pix_bary_v      : in  std_logic_vector(DATA_WIDTH-1 downto 0);
    pix_depth       : in  std_logic_vector(DATA_WIDTH-1 downto 0);

    -- To SM array — one dispatch port broadcast to all SMs; the sm_sel
    -- signal tells the system integration which SM to route ps_valid to
    ps_valid        : out std_logic;
    ps_sm_sel       : out integer range 0 to NUM_SMS-1;  -- Target SM index
    ps_warp_id      : out unsigned(5 downto 0);
    ps_screen_x     : out unsigned(15 downto 0);
    ps_screen_y     : out unsigned(15 downto 0);
    ps_bary_u       : out std_logic_vector(DATA_WIDTH-1 downto 0);
    ps_bary_v       : out std_logic_vector(DATA_WIDTH-1 downto 0);
    ps_depth        : out std_logic_vector(DATA_WIDTH-1 downto 0);
    ps_ready        : in  std_logic_vector(NUM_SMS-1 downto 0);

    -- Performance counter
    quads_dispatched : out unsigned(31 downto 0)
  );
end entity krafgpu_ps_dispatch;

architecture rtl of krafgpu_ps_dispatch is

  -- ===========================================================================
  -- QUAD COLLECTOR
  -- Accumulate 4 pixels before dispatching. Real hardware dispatches 2×2
  -- screen-space quads; we approximate by collecting 4 sequential pixels
  -- (fine rasteriser already emits them in scan order, so 4 consecutive
  -- pixels from the same scanline are always in the same or adjacent quads).
  -- ===========================================================================
  constant QUAD_SIZE : integer := 4;

  type quad_bary_t is array (0 to QUAD_SIZE-1) of
                       std_logic_vector(DATA_WIDTH-1 downto 0);

  signal quad_x     : unsigned(15 downto 0) := (others => '0');
  signal quad_y     : unsigned(15 downto 0) := (others => '0');
  signal quad_bary_u: quad_bary_t := (others => (others => '0'));
  signal quad_bary_v: quad_bary_t := (others => (others => '0'));
  signal quad_depth : quad_bary_t := (others => (others => '0'));
  signal quad_cnt   : integer range 0 to QUAD_SIZE := 0;
  signal quad_ready : std_logic := '0';

  -- ===========================================================================
  -- SM ROUND-ROBIN SELECTOR
  -- ===========================================================================
  signal sm_rr_ptr  : integer range 0 to NUM_SMS-1 := 0;
  signal warp_rr    : integer range 0 to WARPS_PER_SM-1 := 0;

  signal ps_valid_r     : std_logic := '0';
  signal ps_sm_sel_r    : integer range 0 to NUM_SMS-1 := 0;
  signal ps_warp_id_r   : unsigned(5 downto 0) := (others => '0');
  signal ps_screen_x_r  : unsigned(15 downto 0) := (others => '0');
  signal ps_screen_y_r  : unsigned(15 downto 0) := (others => '0');
  signal ps_bary_u_r    : std_logic_vector(DATA_WIDTH-1 downto 0) := (others => '0');
  signal ps_bary_v_r    : std_logic_vector(DATA_WIDTH-1 downto 0) := (others => '0');
  signal ps_depth_r     : std_logic_vector(DATA_WIDTH-1 downto 0) := (others => '0');
  signal quads_disp_r   : unsigned(31 downto 0) := (others => '0');

begin

  ps_valid        <= ps_valid_r;
  ps_sm_sel       <= ps_sm_sel_r;
  ps_warp_id      <= ps_warp_id_r;
  ps_screen_x     <= ps_screen_x_r;
  ps_screen_y     <= ps_screen_y_r;
  ps_bary_u       <= ps_bary_u_r;
  ps_bary_v       <= ps_bary_v_r;
  ps_depth        <= ps_depth_r;
  quads_dispatched<= quads_disp_r;

  -- ===========================================================================
  -- QUAD COLLECTOR PROCESS
  -- ===========================================================================
  p_collect : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        quad_cnt   <= 0;
        quad_ready <= '0';
      else
        quad_ready <= '0';
        if pix_valid = '1' then
          -- Latch anchor coordinates from first pixel of quad
          if quad_cnt = 0 then
            quad_x <= pix_x;
            quad_y <= pix_y;
          end if;
          -- Store per-pixel attributes
          quad_bary_u(quad_cnt) <= pix_bary_u;
          quad_bary_v(quad_cnt) <= pix_bary_v;
          quad_depth(quad_cnt)  <= pix_depth;

          if quad_cnt = QUAD_SIZE - 1 then
            quad_cnt   <= 0;
            quad_ready <= '1';   -- Quad complete, ready to dispatch
          else
            quad_cnt   <= quad_cnt + 1;
          end if;
        end if;
      end if;
    end if;
  end process p_collect;

  -- ===========================================================================
  -- DISPATCH STATE MACHINE
  -- When a quad is ready, find the next available SM (ps_ready = '1') in
  -- round-robin order and fire the dispatch. If no SM is ready, hold the
  -- quad for up to 16 cycles before discarding (back-pressure relief).
  -- ===========================================================================
  p_dispatch : process(clk)
    variable target_sm   : integer range 0 to NUM_SMS-1;
    variable found_sm    : std_logic;
    variable hold_cnt    : integer range 0 to 15 := 0;
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        ps_valid_r   <= '0';
        sm_rr_ptr    <= 0;
        warp_rr      <= 0;
        quads_disp_r <= (others => '0');
        hold_cnt     := 0;
      else
        ps_valid_r <= '0';

        if quad_ready = '1' or hold_cnt > 0 then
          -- Scan from current rr_ptr for a ready SM
          found_sm  := '0';
          target_sm := sm_rr_ptr;

          for offset in 0 to NUM_SMS-1 loop
            if found_sm = '0' then
              target_sm := (sm_rr_ptr + offset) mod NUM_SMS;
              if ps_ready(target_sm) = '1' then
                found_sm := '1';
              end if;
            end if;
          end loop;

          if found_sm = '1' then
            -- Issue dispatch
            ps_valid_r    <= '1';
            ps_sm_sel_r   <= target_sm;
            ps_warp_id_r  <= to_unsigned(warp_rr, 6);
            ps_screen_x_r <= quad_x;
            ps_screen_y_r <= quad_y;
            ps_bary_u_r   <= quad_bary_u(0);   -- Pixel (0,0) of quad
            ps_bary_v_r   <= quad_bary_v(0);
            ps_depth_r    <= quad_depth(0);

            -- Advance RR pointers
            sm_rr_ptr    <= (target_sm + 1) mod NUM_SMS;
            warp_rr      <= (warp_rr + 1) mod WARPS_PER_SM;
            quads_disp_r <= quads_disp_r + 1;
            hold_cnt     := 0;
          else
            -- No SM ready; hold quad for next cycle
            if quad_ready = '1' then
              hold_cnt := 1;
            elsif hold_cnt < 15 then
              hold_cnt := hold_cnt + 1;
            else
              hold_cnt := 0;  -- Drop the quad (all SMs busy for >16 cycles)
            end if;
          end if;
        end if;
      end if;
    end if;
  end process p_dispatch;

end architecture rtl;
