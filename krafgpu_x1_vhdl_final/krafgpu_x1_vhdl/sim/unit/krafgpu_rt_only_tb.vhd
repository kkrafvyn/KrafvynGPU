-- =============================================================================
-- KrafGPU-X1 Unit Test: RT Core (isolated)
-- Project      : KrafGPU-X1 Next-Generation GPU Architecture
-- Author       : Krafvyn Tech / Don
-- Description  : Elaborates and runs one krafgpu_rt_core instance with every
--                input explicitly driven, in isolation from krafgpu_sm.
--                Used during development to confirm the RT core's BVH
--                traversal / Moller-Trumbore intersection FSM is clean on
--                its own, independent of any SM-level integration issue.
-- Run with     : ghdl -r --std=08 --workdir=<workdir> --work=work
--                  krafgpu_rt_only_tb --stop-time=200ns
-- =============================================================================
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_rt_only_tb is
end entity krafgpu_rt_only_tb;

architecture sim of krafgpu_rt_only_tb is
  signal clk    : std_logic := '0';
  signal clk_en : std_logic := '1';
  signal rst_n  : std_logic := '0';
  signal ray_valid : std_logic := '0';
  signal ray_origin_x, ray_origin_y, ray_origin_z : std_logic_vector(31 downto 0) := (others => '0');
  signal ray_dir_x, ray_dir_y, ray_dir_z : std_logic_vector(31 downto 0) := (others => '0');
  signal ray_tmin, ray_tmax : std_logic_vector(31 downto 0) := (others => '0');
  signal ray_warp_id : unsigned(5 downto 0) := (others => '0');
  signal ray_dst_reg : unsigned(7 downto 0) := (others => '0');
  signal bvh_data_valid : std_logic := '0';
  signal bvh_node_data : std_logic_vector(CACHE_LINE_BITS-1 downto 0) := (others => '0');
  signal tri_data_valid : std_logic := '0';
  signal tri_data : std_logic_vector(CACHE_LINE_BITS-1 downto 0) := (others => '0');
  signal mem_req : mem_req_t;
  signal result_valid : std_logic;
  signal result_t, result_u, result_v, result_tri_idx : std_logic_vector(31 downto 0);
  signal result_warp_id : unsigned(5 downto 0);
  signal result_dst_reg : unsigned(7 downto 0);
begin
  clk <= not clk after 1 ns;

  u_rt : entity work.krafgpu_rt_core
    generic map (RT_CORE_ID => 0, SM_ID => 0)
    port map (
      clk => clk, clk_en => clk_en, rst_n => rst_n,
      ray_valid => ray_valid,
      ray_origin_x => ray_origin_x, ray_origin_y => ray_origin_y, ray_origin_z => ray_origin_z,
      ray_dir_x => ray_dir_x, ray_dir_y => ray_dir_y, ray_dir_z => ray_dir_z,
      ray_tmin => ray_tmin, ray_tmax => ray_tmax,
      ray_warp_id => ray_warp_id, ray_dst_reg => ray_dst_reg,
      bvh_data_valid => bvh_data_valid, bvh_node_data => bvh_node_data,
      tri_data_valid => tri_data_valid, tri_data => tri_data,
      mem_req => mem_req,
      result_valid => result_valid, result_t => result_t, result_u => result_u,
      result_v => result_v, result_tri_idx => result_tri_idx,
      result_warp_id => result_warp_id, result_dst_reg => result_dst_reg
    );

  process
  begin
    rst_n <= '0';
    wait for 20 ns;
    rst_n <= '1';
    wait for 50 ns;
    report "RT core (isolated) ran 50ns without error.";
    assert false report "RT-only smoke test complete." severity failure;
    wait;
  end process;
end architecture sim;
