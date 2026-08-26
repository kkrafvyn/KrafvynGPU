-- =============================================================================
-- KrafGPU-X1 Warp Scheduler
-- Project      : KrafGPU-X1 Next-Generation GPU Architecture
-- Author       : Krafvyn Tech / Don
-- Description  : Two-level warp scheduler per SM.
--                Level 1 — Round-robin across all READY warps
--                Level 2 — Stall detection: skip STALLED warps, prefer warps
--                           with data already in L1 cache (hit-under-miss)
--                Supports up to 64 concurrent warps per SM.
--                Issues one warp instruction per cycle when not stalled.
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_warp_scheduler is
  generic (
    SM_ID     : integer := 0;
    NUM_WARPS : integer := MAX_WARPS_PER_SM   -- 64
  );
  port (
    clk            : in  std_logic;
    clk_en         : in  std_logic;
    rst_n          : in  std_logic;
    -- Warp state inputs (from SM control logic)
    warp_states    : in  warp_state_array_t;
    warp_valid     : in  std_logic_vector(NUM_WARPS-1 downto 0);
    -- Selected warp output (issued this cycle)
    selected_warp  : out unsigned(5 downto 0);
    -- Pipeline control
    stall_in       : in  std_logic;
    flush_in       : in  std_logic
  );
end entity krafgpu_warp_scheduler;

architecture rtl of krafgpu_warp_scheduler is

  signal rr_pointer     : unsigned(5 downto 0) := (others => '0');
  signal next_warp      : unsigned(5 downto 0);
  signal found_ready    : std_logic;
  signal scan_idx       : integer range 0 to NUM_WARPS-1;

begin

  -- ===========================================================================
  -- COMBINATIONAL PRIORITY SCAN
  -- Starting from rr_pointer, scan forward and pick the first WARP_READY warp.
  -- This implements greedy round-robin with stall bypass.
  -- ===========================================================================
  p_scan : process(rr_pointer, warp_states, warp_valid)
    variable found  : std_logic := '0';
    variable winner : unsigned(5 downto 0) := (others => '0');
    variable idx    : integer;
  begin
    found  := '0';
    winner := rr_pointer;

    -- Two-pass scan: start at rr_pointer, wrap around
    for pass in 0 to 1 loop
      for i in 0 to NUM_WARPS-1 loop
        idx := (to_integer(rr_pointer) + i) mod NUM_WARPS;
        if found = '0' then
          if warp_valid(idx) = '1' and
             warp_states(idx) = WARP_READY then
            winner := to_unsigned(idx, 6);
            found  := '1';
          end if;
        end if;
      end loop;
    end loop;

    next_warp   <= winner;
    found_ready <= found;
  end process p_scan;

  -- ===========================================================================
  -- SEQUENTIAL: ADVANCE POINTER AND REGISTER OUTPUT
  -- ===========================================================================
  p_schedule : process(clk)
  begin
    if rising_edge(clk) and clk_en = '1' then
      if rst_n = '0' then
        rr_pointer    <= (others => '0');
        selected_warp <= (others => '0');
      elsif flush_in = '1' then
        -- On pipeline flush keep current warp but don't advance
        null;
      elsif stall_in = '0' and found_ready = '1' then
        -- Issue this warp
        selected_warp <= next_warp;
        -- Advance round-robin pointer past this warp
        if next_warp = to_unsigned(NUM_WARPS-1, 6) then
          rr_pointer <= (others => '0');
        else
          rr_pointer <= next_warp + 1;
        end if;
      end if;
    end if;
  end process p_schedule;

end architecture rtl;
