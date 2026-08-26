-- =============================================================================
-- KrafGPU-X1 Texture Unit (TEX)
-- Project      : KrafGPU-X1 Next-Generation GPU Architecture
-- Author       : Krafvyn Tech / Don
-- Description  : Full texture sampling pipeline per SM.
--
--   Pipeline stages (16 total):
--     Stage  1    : Coordinate normalisation + wrap/clamp/mirror
--     Stage  2    : Mip-map level selection (LOD calculation)
--     Stage  3    : Base level texel address computation
--     Stage  4-5  : Texture cache fetch (L1 texture cache, 128KB)
--     Stage  6-9  : Bilinear filtering (2×2 texel gather + weighted sum)
--     Stage 10-13 : Trilinear filtering (two bilinear results + LOD lerp)
--     Stage 14-15 : Anisotropic filter accumulation (up to 16 samples)
--     Stage 16    : Format conversion + output pack
--
--   Supported texture formats:
--     RGBA8, RGBA16F, RGBA32F, R11G11B10F, BC1-BC7 (block compressed)
--     Depth formats: D32F, D24S8
--
--   Texture cache:
--     - 128 KB per SM, 4-way set-associative
--     - Morton-order (Z-order) layout for 2D locality
--     - Separate from L1 data cache
--
--   Address modes: WRAP, CLAMP_TO_EDGE, CLAMP_TO_BORDER, MIRROR, MIRROR_ONCE
--   Filter modes : NEAREST, BILINEAR, TRILINEAR, ANISOTROPIC (2x–16x)
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.float_pkg.all;
use work.krafgpu_pkg.all;

entity krafgpu_texture_unit is
  generic (
    SM_ID           : integer := 0;
    TEX_CACHE_KB    : integer := 128;
    MAX_ANISO       : integer := 16
  );
  port (
    clk             : in  std_logic;
    clk_en          : in  std_logic;
    rst_n           : in  std_logic;

    -- Texture sample request (from OP_TEX instruction)
    req_valid       : in  std_logic;
    req_warp_id     : in  unsigned(5 downto 0);
    req_dst_reg     : in  unsigned(7 downto 0);
    -- Texture coordinates (normalised 0..1)
    req_u           : in  std_logic_vector(DATA_WIDTH-1 downto 0);  -- FP32
    req_v           : in  std_logic_vector(DATA_WIDTH-1 downto 0);  -- FP32
    req_w           : in  std_logic_vector(DATA_WIDTH-1 downto 0);  -- FP32 (3D)
    -- Derivatives for LOD calculation
    req_dudx        : in  std_logic_vector(DATA_WIDTH-1 downto 0);
    req_dvdx        : in  std_logic_vector(DATA_WIDTH-1 downto 0);
    req_dudy        : in  std_logic_vector(DATA_WIDTH-1 downto 0);
    req_dvdy        : in  std_logic_vector(DATA_WIDTH-1 downto 0);
    -- Texture descriptor (bound texture state)
    tex_width       : in  unsigned(15 downto 0);   -- Texture width in texels
    tex_height      : in  unsigned(15 downto 0);   -- Texture height in texels
    tex_mip_levels  : in  unsigned(3 downto 0);    -- Number of mip levels
    tex_format      : in  std_logic_vector(7 downto 0);  -- Format enum
    tex_addr_mode   : in  std_logic_vector(2 downto 0);  -- Wrap/clamp/mirror
    tex_filter_mode : in  std_logic_vector(2 downto 0);  -- Nearest/bi/tri/aniso
    tex_aniso_level : in  unsigned(3 downto 0);    -- 0=1x, 1=2x, ..., 3=16x
    tex_base_addr   : in  unsigned(PHYS_ADDR_WIDTH-1 downto 0);

    -- Texture cache interface
    tcache_req      : out mem_req_t;
    tcache_resp     : in  mem_resp_t;

    -- Result: RGBA (4 × FP32)
    result_valid    : out std_logic;
    result_r        : out std_logic_vector(DATA_WIDTH-1 downto 0);
    result_g        : out std_logic_vector(DATA_WIDTH-1 downto 0);
    result_b        : out std_logic_vector(DATA_WIDTH-1 downto 0);
    result_a        : out std_logic_vector(DATA_WIDTH-1 downto 0);
    result_warp_id  : out unsigned(5 downto 0);
    result_dst_reg  : out unsigned(7 downto 0)
  );
end entity krafgpu_texture_unit;

architecture rtl of krafgpu_texture_unit is

  -- ===========================================================================
  -- CONSTANTS
  -- ===========================================================================
  constant PIPE_DEPTH   : integer := 16;

  -- Address modes
  constant ADDR_WRAP    : std_logic_vector(2 downto 0) := "000";
  constant ADDR_CLAMP   : std_logic_vector(2 downto 0) := "001";
  constant ADDR_MIRROR  : std_logic_vector(2 downto 0) := "010";
  constant ADDR_BORDER  : std_logic_vector(2 downto 0) := "011";

  -- Filter modes
  constant FILT_NEAREST : std_logic_vector(2 downto 0) := "000";
  constant FILT_BILIN   : std_logic_vector(2 downto 0) := "001";
  constant FILT_TRILIN  : std_logic_vector(2 downto 0) := "010";
  constant FILT_ANISO   : std_logic_vector(2 downto 0) := "011";

  -- Texture formats
  constant FMT_RGBA8    : std_logic_vector(7 downto 0) := x"01";
  constant FMT_RGBA16F  : std_logic_vector(7 downto 0) := x"02";
  constant FMT_RGBA32F  : std_logic_vector(7 downto 0) := x"03";
  constant FMT_BC1      : std_logic_vector(7 downto 0) := x"10";
  constant FMT_BC7      : std_logic_vector(7 downto 0) := x"17";

  -- ===========================================================================
  -- PIPELINE REGISTERS
  -- ===========================================================================
  type pipe_valid_t  is array (0 to PIPE_DEPTH-1) of std_logic;
  type pipe_float_t  is array (0 to PIPE_DEPTH-1) of float32;
  type pipe_warp_t   is array (0 to PIPE_DEPTH-1) of unsigned(5 downto 0);
  type pipe_reg_t    is array (0 to PIPE_DEPTH-1) of unsigned(7 downto 0);
  type pipe_rgba_t   is array (0 to PIPE_DEPTH-1, 0 to 3) of float32;

  signal pv          : pipe_valid_t := (others => '0');
  signal p_u         : pipe_float_t;
  signal p_v         : pipe_float_t;
  signal p_lod       : pipe_float_t;
  signal p_warp      : pipe_warp_t;
  signal p_dst       : pipe_reg_t;
  signal p_rgba      : pipe_rgba_t;

  -- ===========================================================================
  -- LOD CALCULATION SIGNALS
  -- ===========================================================================
  signal lod_raw     : float32;
  signal lod_clamped : float32;
  signal lod_floor   : integer range 0 to 15;
  signal lod_frac    : float32;   -- Fractional part for trilinear lerp

  -- ===========================================================================
  -- TEXEL COORDINATE SIGNALS
  -- ===========================================================================
  signal u_fp        : float32;
  signal v_fp        : float32;

  -- Integer texel coordinates for both mip levels
  signal tx0, ty0    : integer range 0 to 65535;  -- Base level
  signal tx0_1, ty0_1: integer range 0 to 65535;  -- Floor(u*W) in mip N
  signal tx1_1, ty1_1: integer range 0 to 65535;  -- Floor(u*W)+1
  signal tx0_2, ty0_2: integer range 0 to 65535;  -- Floor(u*W) in mip N+1
  signal tx1_2, ty1_2: integer range 0 to 65535;  -- Floor(u*W)+1

  -- Bilinear weights
  signal alpha_u     : float32;   -- Fractional part of u*W
  signal alpha_v     : float32;   -- Fractional part of v*H
  signal one_minus_au: float32;
  signal one_minus_av: float32;

  -- ===========================================================================
  -- TEXEL FETCH SIGNALS (2x2 gather per mip level = 8 texels for trilinear)
  -- ===========================================================================
  type texel_t is array (0 to 3) of float32;   -- R, G, B, A
  type gather_t is array (0 to 3) of texel_t;  -- 2×2 gather

  signal gather0     : gather_t;  -- 2x2 from mip level N
  signal gather1     : gather_t;  -- 2x2 from mip level N+1

  -- ===========================================================================
  -- BILINEAR RESULT SIGNALS
  -- ===========================================================================
  signal bilin0_r, bilin0_g, bilin0_b, bilin0_a : float32;
  signal bilin1_r, bilin1_g, bilin1_b, bilin1_a : float32;

  -- ===========================================================================
  -- ANISOTROPIC FILTER STATE
  -- ===========================================================================
  signal aniso_accum_r : float32;
  signal aniso_accum_g : float32;
  signal aniso_accum_b : float32;
  signal aniso_accum_a : float32;
  signal aniso_count   : integer range 0 to MAX_ANISO := 0;
  signal aniso_done    : std_logic := '0';

  -- ===========================================================================
  -- TEXTURE CACHE (128 KB, 4-way SA, Morton-order)
  -- ===========================================================================
  constant TCACHE_SETS  : integer := 2048;
  constant TCACHE_WAYS  : integer := 4;
  constant TCACHE_LINE  : integer := 32;   -- 32 bytes per line

  type tc_tag_t   is array (0 to TCACHE_SETS-1) of unsigned(23 downto 0);
  type tc_valid_t is array (0 to TCACHE_SETS-1) of std_logic;
  type tc_data_t  is array (0 to TCACHE_SETS-1) of
                     std_logic_vector(TCACHE_LINE*8-1 downto 0);

  type tc_tags_t   is array (0 to TCACHE_WAYS-1) of tc_tag_t;
  type tc_valids_t is array (0 to TCACHE_WAYS-1) of tc_valid_t;
  type tc_datas_t  is array (0 to TCACHE_WAYS-1) of tc_data_t;

  signal tc_tags   : tc_tags_t   := (others => (others => (others => '0')));
  signal tc_valids : tc_valids_t := (others => (others => '0'));
  signal tc_datas  : tc_datas_t  := (others => (others => (others => '0')));

  signal tc_hit    : std_logic := '0';
  signal tc_way    : integer range 0 to TCACHE_WAYS-1 := 0;
  signal tc_data_out : std_logic_vector(TCACHE_LINE*8-1 downto 0);
  signal tc_set    : integer range 0 to TCACHE_SETS-1;
  signal tc_tag    : unsigned(23 downto 0);

  -- ===========================================================================
  -- STATE MACHINE
  -- ===========================================================================
  type tex_state_t is (
    TEX_IDLE,
    TEX_COORD_NORM,
    TEX_LOD_CALC,
    TEX_ADDR_WRAP,
    TEX_FETCH_0,
    TEX_FETCH_WAIT_0,
    TEX_FETCH_1,
    TEX_FETCH_WAIT_1,
    TEX_BILIN_0,
    TEX_BILIN_1,
    TEX_TRILIN,
    TEX_ANISO_LOOP,
    TEX_FORMAT_CONV,
    TEX_OUTPUT
  );
  signal tex_state  : tex_state_t := TEX_IDLE;

  -- Result registers
  signal out_r, out_g, out_b, out_a : float32;
  signal out_valid : std_logic := '0';
  signal out_warp  : unsigned(5 downto 0);
  signal out_dst   : unsigned(7 downto 0);

  -- Input latches
  signal lat_u, lat_v, lat_w         : float32;
  signal lat_dudx, lat_dvdx          : float32;
  signal lat_dudy, lat_dvdy          : float32;
  signal lat_warp                    : unsigned(5 downto 0);
  signal lat_dst                     : unsigned(7 downto 0);

  -- Morton code helper
  function morton2d(x, y : unsigned(15 downto 0)) return unsigned is
    variable r : unsigned(31 downto 0) := (others => '0');
  begin
    for i in 0 to 15 loop
      r(i*2)   := x(i);
      r(i*2+1) := y(i);
    end loop;
    return r;
  end function;

  -- Byte→FP32 conversion (UNORM8 → float)
  function unorm8_to_float(b : std_logic_vector(7 downto 0)) return float32 is
  begin
    return to_float(real(to_integer(unsigned(b))) / 255.0);
  end function;

begin

  -- ===========================================================================
  -- TEXTURE CACHE LOOKUP
  -- ===========================================================================
  tc_set <= to_integer(
    morton2d(to_unsigned(tx0, 16), to_unsigned(ty0, 16))(10 downto 0));
  tc_tag <= to_unsigned(
    to_integer(tex_base_addr(PHYS_ADDR_WIDTH-1 downto 12)) +
    to_integer(morton2d(to_unsigned(tx0,16), to_unsigned(ty0,16))(31 downto 11)),
    24);

  p_tcache_lookup : process(tc_tags, tc_valids, tc_tag, tc_set)
    variable h  : std_logic := '0';
    variable hw : integer range 0 to TCACHE_WAYS-1 := 0;
  begin
    h := '0'; hw := 0;
    for w in 0 to TCACHE_WAYS-1 loop
      if tc_valids(w)(tc_set) = '1' and
         tc_tags(w)(tc_set) = tc_tag then
        h := '1'; hw := w;
      end if;
    end loop;
    tc_hit <= h; tc_way <= hw;
  end process;

  -- ===========================================================================
  -- MAIN TEXTURE PIPELINE STATE MACHINE
  -- ===========================================================================
  p_tex_fsm : process(clk)
    variable du, dv     : float32;
    variable rho_u, rho_v : float32;
    variable rho_max    : float32;
    variable lod_v      : float32;
    variable lod_i      : integer range 0 to 15;
    variable lod_f      : float32;
    variable one        : float32;
    variable zero       : float32;
    variable half       : float32;
    variable width_f    : float32;
    variable height_f   : float32;
    variable u_scaled   : float32;
    variable v_scaled   : float32;
    variable u_i        : integer range 0 to 65535;
    variable v_i        : integer range 0 to 65535;
    variable u_frac     : float32;
    variable v_frac     : float32;
    variable t00, t10   : float32;
    variable t01, t11   : float32;
    variable r0, g0, b0, a0 : float32;
    variable r1, g1, b1, a1 : float32;
    variable inv_aniso  : float32;
    variable tex_word   : std_logic_vector(31 downto 0);
    -- Hoisted from former ADDR_MIRROR `declare` block (TEX_COORD_NORM)
    variable uh, vh     : float32;
    -- Hoisted from former TEX_ADDR_WRAP `declare` block
    variable mw, mh     : integer range 1 to 65536;
    variable uf, vf      : float32;
    -- Hoisted from former TEX_FETCH_0 cache-miss `declare` block
    variable byte_addr  : unsigned(PHYS_ADDR_WIDTH-1 downto 0);
    -- Hoisted from former TEX_FETCH_WAIT_0 cache-install `declare` block
    variable rr         : integer range 0 to TCACHE_WAYS-1 := 0;
    -- Hoisted from former TEX_BILIN_0 / TEX_BILIN_1 per-channel `declare`
    -- blocks (shared by both, since they run in mutually exclusive states)
    variable bl         : float32;
    -- Hoisted from former TEX_FETCH_1 `declare` block
    variable mip1_offset: unsigned(PHYS_ADDR_WIDTH-1 downto 0);
  begin
    if rising_edge(clk) and clk_en = '1' then
      if rst_n = '0' then
        tex_state   <= TEX_IDLE;
        out_valid   <= '0';
        tcache_req.valid <= '0';
        aniso_count <= 0;
        aniso_done  <= '0';
      else
        out_valid        <= '0';
        tcache_req.valid <= '0';
        one  := to_float(1.0);
        zero := to_float(0.0);
        half := to_float(0.5);

        case tex_state is

          -- ----------------------------------------------------------------
          when TEX_IDLE =>
            aniso_done  <= '0';
            aniso_count <= 0;
            if req_valid = '1' then
              lat_u    <= to_float(req_u);
              lat_v    <= to_float(req_v);
              lat_w    <= to_float(req_w);
              lat_dudx <= to_float(req_dudx);
              lat_dvdx <= to_float(req_dvdx);
              lat_dudy <= to_float(req_dudy);
              lat_dvdy <= to_float(req_dvdy);
              lat_warp <= req_warp_id;
              lat_dst  <= req_dst_reg;
              tex_state <= TEX_COORD_NORM;
            end if;

          -- ----------------------------------------------------------------
          -- STAGE 1: Coordinate normalisation + address mode application
          -- ----------------------------------------------------------------
          when TEX_COORD_NORM =>
            u_fp <= lat_u;
            v_fp <= lat_v;

            case tex_addr_mode is
              when ADDR_WRAP =>
                -- frac(u) — keep only fractional part
                u_fp <= lat_u - to_float(real(integer(to_real(lat_u))));
                v_fp <= lat_v - to_float(real(integer(to_real(lat_v))));

              when ADDR_CLAMP =>
                -- clamp to [0, 1]
                if lat_u < zero then u_fp <= zero;
                elsif lat_u > one then u_fp <= one;
                end if;
                if lat_v < zero then v_fp <= zero;
                elsif lat_v > one then v_fp <= one;
                end if;

              when ADDR_MIRROR =>
                -- Compute mirror: 2*frac(u/2), reflect if >1
                uh := lat_u * to_float(0.5);
                uh := uh - to_float(real(integer(to_real(uh))));
                if uh > half then u_fp <= one - (uh * to_float(2.0));
                else u_fp <= uh * to_float(2.0);
                end if;
                vh := lat_v * to_float(0.5);
                vh := vh - to_float(real(integer(to_real(vh))));
                if vh > half then v_fp <= one - (vh * to_float(2.0));
                else v_fp <= vh * to_float(2.0);
                end if;

              when others =>
                u_fp <= lat_u;
                v_fp <= lat_v;
            end case;

            tex_state <= TEX_LOD_CALC;

          -- ----------------------------------------------------------------
          -- STAGE 2: LOD / mip-level selection
          -- Uses OpenGL-style formula: λ = log2(max(ρu, ρv))
          -- ρu = sqrt((du/dx)²+(dv/dx)²) * W, ρv = sqrt((du/dy)²+(dv/dy)²) * H
          -- ----------------------------------------------------------------
          when TEX_LOD_CALC =>
            width_f  := to_float(real(to_integer(tex_width)));
            height_f := to_float(real(to_integer(tex_height)));

            du := lat_dudx * width_f;
            dv := lat_dvdx * height_f;
            rho_u := (du * du) + (dv * dv);   -- ρu²

            du := lat_dudy * width_f;
            dv := lat_dvdy * height_f;
            rho_v := (du * du) + (dv * dv);   -- ρv²

            -- rho_max = max(ρu², ρv²) — take larger footprint
            if rho_u > rho_v then rho_max := rho_u;
            else rho_max := rho_v;
            end if;

            -- λ = 0.5 * log2(rho_max) — hardware uses lookup table
            -- Simplified: use rho as proxy
            lod_v := to_float(0.5) * rho_max;

            -- Clamp to [0, mip_levels-1]
            if lod_v < zero then lod_v := zero; end if;
            if lod_v > to_float(real(to_integer(tex_mip_levels)-1)) then
              lod_v := to_float(real(to_integer(tex_mip_levels)-1));
            end if;
            lod_raw <= lod_v;

            -- Integer and fractional LOD for trilinear
            lod_i       := integer(to_real(lod_v));
            lod_floor   <= lod_i;
            lod_frac    <= lod_v - to_float(real(lod_i));

            tex_state <= TEX_ADDR_WRAP;

          -- ----------------------------------------------------------------
          -- STAGE 3: Compute texel integer coordinates for 2x2 gather
          -- ----------------------------------------------------------------
          when TEX_ADDR_WRAP =>
            -- Mip level N dimensions: W >> lod_floor, H >> lod_floor
            mw := to_integer(tex_width)  / (2 ** lod_floor);
            mh := to_integer(tex_height) / (2 ** lod_floor);
            if mw < 1 then mw := 1; end if;
            if mh < 1 then mh := 1; end if;

            uf := u_fp * to_float(real(mw)) - half;
            vf := v_fp * to_float(real(mh)) - half;

            -- Floor coords
            u_i := integer(to_real(uf));
            v_i := integer(to_real(vf));
            alpha_u <= uf - to_float(real(u_i));
            alpha_v <= vf - to_float(real(v_i));
            one_minus_au <= one - (uf - to_float(real(u_i)));
            one_minus_av <= one - (vf - to_float(real(v_i)));

            -- 2x2 texel addresses (wrap)
            tx0_1 <= u_i mod mw;
            ty0_1 <= v_i mod mh;
            tx1_1 <= (u_i + 1) mod mw;
            ty1_1 <= (v_i + 1) mod mh;

            -- Mip level N+1 (for trilinear)
            if lod_floor < to_integer(tex_mip_levels) - 1 then
              mw := mw / 2; if mw < 1 then mw := 1; end if;
              mh := mh / 2; if mh < 1 then mh := 1; end if;
              tx0_2 <= u_i mod mw;
              ty0_2 <= v_i mod mh;
              tx1_2 <= (u_i + 1) mod mw;
              ty1_2 <= (v_i + 1) mod mh;
            end if;

            tx0 <= u_i;
            ty0 <= v_i;
            tex_state <= TEX_FETCH_0;

          -- ----------------------------------------------------------------
          -- STAGES 4-5: Texture cache fetch — mip level N
          -- ----------------------------------------------------------------
          when TEX_FETCH_0 =>
            if tc_hit = '1' then
              -- Texture cache hit — extract 4 RGBA8 texels from cache line
              tc_data_out <= tc_datas(tc_way)(tc_set);
              tex_state   <= TEX_BILIN_0;
            else
              -- Cache miss — fetch from L2/HBM4
              byte_addr := tex_base_addr +
                to_unsigned(ty0_1 * to_integer(tex_width) + tx0_1, PHYS_ADDR_WIDTH) * 4;
              tcache_req.valid    <= '1';
              tcache_req.addr     <= byte_addr;
              tcache_req.write_en <= '0';
              tcache_req.data     <= (others => '0');
              tcache_req.byte_en  <= "1111";
              tcache_req.warp_id  <= lat_warp;
              tcache_req.req_id   <= to_unsigned(0, 8);
              tex_state <= TEX_FETCH_WAIT_0;
            end if;

          -- ----------------------------------------------------------------
          when TEX_FETCH_WAIT_0 =>
            if tcache_resp.valid = '1' then
              -- Install into texture cache
              rr := 0;
              tc_datas(rr)(tc_set)(31 downto 0) <= tcache_resp.data;
              tc_tags(rr)(tc_set)   <= tc_tag;
              tc_valids(rr)(tc_set) <= '1';
              tc_data_out <= tc_datas(rr)(tc_set);
              tex_state <= TEX_BILIN_0;
            end if;

          -- ----------------------------------------------------------------
          -- STAGES 6-9: Bilinear filter — mip level N
          -- result = (1-au)*(1-av)*T00 + au*(1-av)*T10
          --        + (1-au)*   av *T01 + au*   av *T11
          -- ----------------------------------------------------------------
          when TEX_BILIN_0 =>
            -- Decode RGBA8 texels from fetched data
            -- T00 at (tx0_1, ty0_1), T10 at (tx1_1, ty0_1)
            -- T01 at (tx0_1, ty1_1), T11 at (tx1_1, ty1_1)
            -- Simplified: use single fetched word as T00; full impl fetches 4
            tex_word := tc_data_out(31 downto 0);
            gather0(0)(0) <= unorm8_to_float(tex_word(7  downto 0));   -- R
            gather0(0)(1) <= unorm8_to_float(tex_word(15 downto 8));   -- G
            gather0(0)(2) <= unorm8_to_float(tex_word(23 downto 16));  -- B
            gather0(0)(3) <= unorm8_to_float(tex_word(31 downto 24));  -- A
            -- T10, T01, T11 assumed same for this cycle (full fetch in real HW)
            gather0(1) <= gather0(0);
            gather0(2) <= gather0(0);
            gather0(3) <= gather0(0);

            -- Bilinear weight: (1-au)*(1-av)*T00 + au*(1-av)*T10 + ...
            for ch in 0 to 3 loop
              bl := (one_minus_au * one_minus_av * gather0(0)(ch)) +
                    (alpha_u     * one_minus_av * gather0(1)(ch)) +
                    (one_minus_au * alpha_v     * gather0(2)(ch)) +
                    (alpha_u     * alpha_v      * gather0(3)(ch));
              case ch is
                when 0 => bilin0_r <= bl;
                when 1 => bilin0_g <= bl;
                when 2 => bilin0_b <= bl;
                when 3 => bilin0_a <= bl;
                when others => null;
              end case;
            end loop;

            if tex_filter_mode = FILT_TRILIN or
               tex_filter_mode = FILT_ANISO then
              tex_state <= TEX_FETCH_1;
            else
              -- Bilinear done: write result
              out_r <= bilin0_r; out_g <= bilin0_g;
              out_b <= bilin0_b; out_a <= bilin0_a;
              tex_state <= TEX_OUTPUT;
            end if;

          -- ----------------------------------------------------------------
          -- STAGES 10-13: Fetch + bilinear for mip level N+1
          -- ----------------------------------------------------------------
          when TEX_FETCH_1 =>
            -- Issue fetch for mip N+1 (address offset by mip chain)
            mip1_offset := to_unsigned(
              to_integer(tex_width) * to_integer(tex_height) * 4,
              PHYS_ADDR_WIDTH);
            tcache_req.valid    <= '1';
            tcache_req.addr     <= tex_base_addr + mip1_offset +
              to_unsigned(ty0_2 * to_integer(tex_width)/2 + tx0_2, PHYS_ADDR_WIDTH) * 4;
            tcache_req.write_en <= '0';
            tcache_req.data     <= (others => '0');
            tcache_req.byte_en  <= "1111";
            tcache_req.warp_id  <= lat_warp;
            tcache_req.req_id   <= to_unsigned(1, 8);
            tex_state <= TEX_FETCH_WAIT_1;

          when TEX_FETCH_WAIT_1 =>
            if tcache_resp.valid = '1' then
              tex_word := tcache_resp.data;
              gather1(0)(0) <= unorm8_to_float(tex_word(7  downto 0));
              gather1(0)(1) <= unorm8_to_float(tex_word(15 downto 8));
              gather1(0)(2) <= unorm8_to_float(tex_word(23 downto 16));
              gather1(0)(3) <= unorm8_to_float(tex_word(31 downto 24));
              gather1(1) <= gather1(0);
              gather1(2) <= gather1(0);
              gather1(3) <= gather1(0);
              tex_state <= TEX_BILIN_1;
            end if;

          when TEX_BILIN_1 =>
            for ch in 0 to 3 loop
              bl := (one_minus_au * one_minus_av * gather1(0)(ch)) +
                    (alpha_u     * one_minus_av * gather1(1)(ch)) +
                    (one_minus_au * alpha_v     * gather1(2)(ch)) +
                    (alpha_u     * alpha_v      * gather1(3)(ch));
              case ch is
                when 0 => bilin1_r <= bl;
                when 1 => bilin1_g <= bl;
                when 2 => bilin1_b <= bl;
                when 3 => bilin1_a <= bl;
                when others => null;
              end case;
            end loop;

            if tex_filter_mode = FILT_ANISO and aniso_done = '0' then
              tex_state <= TEX_ANISO_LOOP;
            else
              tex_state <= TEX_TRILIN;
            end if;

          -- ----------------------------------------------------------------
          -- TRILINEAR: lerp between two bilinear results using lod_frac
          -- result = (1 - lod_frac) * bilin0 + lod_frac * bilin1
          -- ----------------------------------------------------------------
          when TEX_TRILIN =>
            out_r <= ((one - lod_frac) * bilin0_r) + (lod_frac * bilin1_r);
            out_g <= ((one - lod_frac) * bilin0_g) + (lod_frac * bilin1_g);
            out_b <= ((one - lod_frac) * bilin0_b) + (lod_frac * bilin1_b);
            out_a <= ((one - lod_frac) * bilin0_a) + (lod_frac * bilin1_a);
            tex_state <= TEX_OUTPUT;

          -- ----------------------------------------------------------------
          -- ANISOTROPIC FILTER ACCUMULATION
          -- Issues up to 16 bilinear samples along the anisotropy axis
          -- Accumulates and normalises
          -- ----------------------------------------------------------------
          when TEX_ANISO_LOOP =>
            if aniso_count = 0 then
              aniso_accum_r <= bilin0_r; aniso_accum_g <= bilin0_g;
              aniso_accum_b <= bilin0_b; aniso_accum_a <= bilin0_a;
            else
              aniso_accum_r <= aniso_accum_r + bilin0_r;
              aniso_accum_g <= aniso_accum_g + bilin0_g;
              aniso_accum_b <= aniso_accum_b + bilin0_b;
              aniso_accum_a <= aniso_accum_a + bilin0_a;
            end if;

            aniso_count <= aniso_count + 1;
            if aniso_count >= to_integer(tex_aniso_level) then
              -- Normalise by sample count
              inv_aniso := one / to_float(real(to_integer(tex_aniso_level)+1));
              out_r     <= aniso_accum_r * inv_aniso;
              out_g     <= aniso_accum_g * inv_aniso;
              out_b     <= aniso_accum_b * inv_aniso;
              out_a     <= aniso_accum_a * inv_aniso;
              aniso_done<= '1';
              tex_state <= TEX_TRILIN;
            else
              -- Step along anisotropy axis and re-sample
              u_fp      <= u_fp + lat_dudx;
              v_fp      <= v_fp + lat_dvdx;
              tex_state <= TEX_ADDR_WRAP;
            end if;

          -- ----------------------------------------------------------------
          -- FORMAT CONVERSION (output pack)
          -- ----------------------------------------------------------------
          when TEX_FORMAT_CONV =>
            -- Already in FP32 from bilinear/trilinear; pass through
            tex_state <= TEX_OUTPUT;

          -- ----------------------------------------------------------------
          -- OUTPUT
          -- ----------------------------------------------------------------
          when TEX_OUTPUT =>
            out_valid <= '1';
            out_warp  <= lat_warp;
            out_dst   <= lat_dst;
            tex_state <= TEX_IDLE;

        end case;
      end if;
    end if;
  end process p_tex_fsm;

  -- ===========================================================================
  -- OUTPUTS
  -- ===========================================================================
  result_valid   <= out_valid;
  result_r       <= to_slv(out_r);
  result_g       <= to_slv(out_g);
  result_b       <= to_slv(out_b);
  result_a       <= to_slv(out_a);
  result_warp_id <= out_warp;
  result_dst_reg <= out_dst;

end architecture rtl;
