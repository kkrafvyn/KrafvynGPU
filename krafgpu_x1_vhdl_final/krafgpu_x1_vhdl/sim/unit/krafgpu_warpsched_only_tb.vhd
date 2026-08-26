-- =============================================================================
-- KrafGPU-X1 Unit Test: Warp Scheduler (isolated)
-- Project      : KrafGPU-X1 Next-Generation GPU Architecture
-- Author       : Krafvyn Tech / Don
-- Description  : Elaborates and runs one krafgpu_warp_scheduler instance
--                with warp_states/warp_valid deliberately left
--                uninitialised (matching krafgpu_sm's exact declaration
--                style), in isolation from krafgpu_sm. Confirms the
--                round-robin priority scan does not itself produce a
--                NUMERIC_STD.TO_INTEGER metavalue warning even with
--                undriven inputs -- this was used to rule the scheduler out
--                as the source of a one-time time-0 warning eventually
--                traced to a dead concurrent assignment in krafgpu_sm.vhd
--                (since removed; see that file's revision history).
-- Run with     : ghdl -r --std=08 --workdir=<workdir> --work=work
--                  krafgpu_warpsched_only_tb --stop-time=200ns
-- =============================================================================
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_warpsched_only_tb is
end entity krafgpu_warpsched_only_tb;

architecture sim of krafgpu_warpsched_only_tb is
  signal clk    : std_logic := '0';
  signal clk_en : std_logic := '1';
  signal rst_n  : std_logic := '0';
  signal warp_states : warp_state_array_t;          -- NO default init (matches SM)
  signal warp_valid  : std_logic_vector(MAX_WARPS_PER_SM-1 downto 0);  -- NO default init (matches SM)
  signal selected_warp : unsigned(5 downto 0);
  signal stall_in : std_logic := '0';
  signal flush_in : std_logic := '0';
begin
  clk <= not clk after 1 ns;

  u_ws : entity work.krafgpu_warp_scheduler
    generic map (SM_ID => 0, NUM_WARPS => MAX_WARPS_PER_SM)
    port map (
      clk => clk, clk_en => clk_en, rst_n => rst_n,
      warp_states => warp_states, warp_valid => warp_valid,
      selected_warp => selected_warp,
      stall_in => stall_in, flush_in => flush_in
    );

  process
  begin
    rst_n <= '0';
    wait for 20 ns;
    rst_n <= '1';
    wait for 50 ns;
    report "Warp scheduler (isolated, uninitialised inputs) ran 50ns without error.";
    assert false report "Warp-sched-only smoke test complete." severity failure;
    wait;
  end process;
end architecture sim;
