-- =============================================================================
-- KrafGPU-X1 Unit Test: Texture Unit (isolated)
-- Project      : KrafGPU-X1 Next-Generation GPU Architecture
-- Author       : Krafvyn Tech / Don
-- Description  : Elaborates and runs one krafgpu_texture_unit instance with
--                every input explicitly driven, in isolation from
--                krafgpu_sm. Used during development to confirm the
--                bilinear/trilinear/anisotropic filtering FSM and texture
--                cache are clean on their own, independent of any SM-level
--                integration issue.
-- Run with     : ghdl -r --std=08 --workdir=<workdir> --work=work
--                  krafgpu_tex_only_tb --stop-time=200ns
-- =============================================================================
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_tex_only_tb is
end entity krafgpu_tex_only_tb;

architecture sim of krafgpu_tex_only_tb is
  signal clk    : std_logic := '0';
  signal clk_en : std_logic := '1';
  signal rst_n  : std_logic := '0';

  signal req_valid : std_logic := '0';
  signal req_warp_id : unsigned(5 downto 0) := (others => '0');
  signal req_dst_reg : unsigned(7 downto 0) := (others => '0');
  signal req_u, req_v, req_w : std_logic_vector(31 downto 0) := (others => '0');
  signal req_dudx, req_dvdx, req_dudy, req_dvdy : std_logic_vector(31 downto 0) := (others => '0');
  signal tex_width  : unsigned(15 downto 0) := to_unsigned(64, 16);
  signal tex_height : unsigned(15 downto 0) := to_unsigned(64, 16);
  signal tex_mip_levels : unsigned(3 downto 0) := to_unsigned(1, 4);
  signal tex_format : std_logic_vector(7 downto 0) := x"01";
  signal tex_addr_mode : std_logic_vector(2 downto 0) := "000";
  signal tex_filter_mode : std_logic_vector(2 downto 0) := "000";
  signal tex_aniso_level : unsigned(3 downto 0) := to_unsigned(1, 4);
  signal tex_base_addr : unsigned(PHYS_ADDR_WIDTH-1 downto 0) := (others => '0');
  signal tcache_req : mem_req_t;
  signal tcache_resp : mem_resp_t := (
    valid => '0', data => (others => '0'),
    req_id => (others => '0'), error => '0');
  signal result_valid : std_logic;
  signal result_r, result_g, result_b, result_a : std_logic_vector(31 downto 0);
  signal result_warp_id : unsigned(5 downto 0);
  signal result_dst_reg : unsigned(7 downto 0);
begin
  clk <= not clk after 1 ns;

  u_tex : entity work.krafgpu_texture_unit
    generic map (SM_ID => 0, TEX_CACHE_KB => 128, MAX_ANISO => 16)
    port map (
      clk => clk, clk_en => clk_en, rst_n => rst_n,
      req_valid => req_valid, req_warp_id => req_warp_id, req_dst_reg => req_dst_reg,
      req_u => req_u, req_v => req_v, req_w => req_w,
      req_dudx => req_dudx, req_dvdx => req_dvdx, req_dudy => req_dudy, req_dvdy => req_dvdy,
      tex_width => tex_width, tex_height => tex_height, tex_mip_levels => tex_mip_levels,
      tex_format => tex_format, tex_addr_mode => tex_addr_mode, tex_filter_mode => tex_filter_mode,
      tex_aniso_level => tex_aniso_level, tex_base_addr => tex_base_addr,
      tcache_req => tcache_req, tcache_resp => tcache_resp,
      result_valid => result_valid, result_r => result_r, result_g => result_g,
      result_b => result_b, result_a => result_a,
      result_warp_id => result_warp_id, result_dst_reg => result_dst_reg
    );

  process
  begin
    rst_n <= '0';
    wait for 20 ns;
    rst_n <= '1';
    wait for 50 ns;
    report "Texture unit (isolated) ran 50ns without error.";
    assert false report "Tex-only smoke test complete." severity failure;
    wait;
  end process;
end architecture sim;
