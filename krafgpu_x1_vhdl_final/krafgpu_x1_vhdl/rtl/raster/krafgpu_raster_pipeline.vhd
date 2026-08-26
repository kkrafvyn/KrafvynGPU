-- =============================================================================
-- KrafGPU-X1 Rasterisation Pipeline
-- Project      : KrafGPU-X1 Next-Generation GPU Architecture
-- Author       : Krafvyn Tech / Don
-- Description  : Fixed-function rasterisation pipeline stages:
--
--   Stage 1 — Triangle Setup
--     Converts 3 clip-space vertices → screen-space, computes edge equations
--     and area (1/2W determinant). Clips against viewport.
--
--   Stage 2 — Tile Binning
--     Divides screen into 16×16 pixel tiles.
--     Each triangle is assigned to all overlapping tiles (coarse raster).
--     Output: per-tile triangle list.
--
--   Stage 3 — Fine Rasterisation (pixel coverage)
--     Iterates pixels within tile, evaluates 3 edge functions at each pixel.
--     Generates coverage mask (4×4 super-sample grid for MSAA 4x/8x/16x).
--
--   Stage 4 — Attribute Interpolation
--     Perspective-correct interpolation of:
--       - Depth Z (for depth test)
--       - 1/W (for perspective division)
--       - Vertex attributes: colour, texcoord, normals (up to 16 attributes)
--
--   Stage 5 — Early-Z / Hi-Z Test
--     Hierarchical depth test (Hi-Z): 8×8 tile compressed depth rejects
--     occluded triangles before pixel shader dispatch.
--     Early-Z: per-pixel depth compare + update against depth buffer.
--
--   Stage 6 — Pixel Shader Dispatch
--     Packages surviving pixels into 2×2 quads, creates warps,
--     dispatches to SM for fragment shader execution.
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.float_pkg.all;
use work.krafgpu_pkg.all;

-- =============================================================================
-- Triangle Setup
-- =============================================================================
entity krafgpu_triangle_setup is
  port (
    clk             : in  std_logic;
    rst_n           : in  std_logic;

    -- Input: 3 vertices in clip space (Xc,Yc,Zc,Wc) × 3
    tri_valid       : in  std_logic;
    v0_x, v0_y, v0_z, v0_w : in std_logic_vector(DATA_WIDTH-1 downto 0);
    v1_x, v1_y, v1_z, v1_w : in std_logic_vector(DATA_WIDTH-1 downto 0);
    v2_x, v2_y, v2_z, v2_w : in std_logic_vector(DATA_WIDTH-1 downto 0);

    -- Viewport transform parameters
    vp_x       : in  unsigned(15 downto 0);  -- Viewport X origin (pixels)
    vp_y       : in  unsigned(15 downto 0);  -- Viewport Y origin
    vp_width   : in  unsigned(15 downto 0);  -- Viewport width
    vp_height  : in  unsigned(15 downto 0);  -- Viewport height

    -- Output: screen-space edge equations + setup data for rasteriser
    setup_valid : out std_logic;
    -- Edge equation coefficients: edge_i(x,y) = A_i*x + B_i*y + C_i
    e0_a, e0_b, e0_c : out signed(31 downto 0);
    e1_a, e1_b, e1_c : out signed(31 downto 0);
    e2_a, e2_b, e2_c : out signed(31 downto 0);
    -- Screen-space bounding box
    bbox_x0, bbox_y0  : out unsigned(15 downto 0);
    bbox_x1, bbox_y1  : out unsigned(15 downto 0);
    -- Reciprocal area (for barycentric interpolation)
    inv_area          : out std_logic_vector(DATA_WIDTH-1 downto 0);
    -- Depth interpolation deltas
    dzdx, dzdy        : out std_logic_vector(DATA_WIDTH-1 downto 0);
    -- Back-face cull flag
    backface          : out std_logic
  );
end entity krafgpu_triangle_setup;

architecture rtl of krafgpu_triangle_setup is

  type setup_state_t is (SU_IDLE, SU_NDC, SU_SCREEN, SU_EDGE, SU_BBOX, SU_DONE);
  signal su_state : setup_state_t := SU_IDLE;

  -- Vertex positions in NDC and screen space
  signal sx0, sy0, sz0 : float32;
  signal sx1, sy1, sz1 : float32;
  signal sx2, sy2, sz2 : float32;

  -- Edge vectors
  signal dx01, dy01 : signed(31 downto 0);
  signal dx12, dy12 : signed(31 downto 0);
  signal dx20, dy20 : signed(31 downto 0);

  signal area_2x    : signed(31 downto 0);
  signal out_valid_r: std_logic := '0';

begin

  p_setup : process(clk)
    variable vp_half_w : float32;
    variable vp_half_h : float32;
    variable w0, w1, w2 : float32;
    variable ndc_x0, ndc_y0 : float32;
    variable ndc_x1, ndc_y1 : float32;
    variable ndc_x2, ndc_y2 : float32;
    variable sc_x0, sc_y0 : integer range -32768 to 32767;
    variable sc_x1, sc_y1 : integer range -32768 to 32767;
    variable sc_x2, sc_y2 : integer range -32768 to 32767;
    variable area_v   : integer;
    variable bx0, by0, bx1, by1 : integer range 0 to 65535;
    variable inv_a    : float32;
    variable dzdx_v, dzdy_v : float32;
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        su_state    <= SU_IDLE;
        out_valid_r <= '0';
        setup_valid <= '0';
      else
        setup_valid <= '0';

        case su_state is
          when SU_IDLE =>
            out_valid_r <= '0';
            if tri_valid = '1' then
              su_state <= SU_NDC;
            end if;

          when SU_NDC =>
            -- Perspective division: NDC = clip / W
            w0 := to_float(v0_w);
            w1 := to_float(v1_w);
            w2 := to_float(v2_w);

            -- Guard against W=0 (degenerate)
            if w0 = to_float(0.0) then w0 := to_float(0.0001); end if;
            if w1 = to_float(0.0) then w1 := to_float(0.0001); end if;
            if w2 = to_float(0.0) then w2 := to_float(0.0001); end if;

            ndc_x0 := to_float(v0_x) / w0;
            ndc_y0 := to_float(v0_y) / w0;
            ndc_x1 := to_float(v1_x) / w1;
            ndc_y1 := to_float(v1_y) / w1;
            ndc_x2 := to_float(v2_x) / w2;
            ndc_y2 := to_float(v2_y) / w2;

            -- Store Z (already divided)
            sz0 <= to_float(v0_z) / w0;
            sz1 <= to_float(v1_z) / w1;
            sz2 <= to_float(v2_z) / w2;

            -- Store for next stage
            sx0 <= ndc_x0; sy0 <= ndc_y0;
            sx1 <= ndc_x1; sy1 <= ndc_y1;
            sx2 <= ndc_x2; sy2 <= ndc_y2;

            su_state <= SU_SCREEN;

          when SU_SCREEN =>
            -- NDC → screen: x_s = (ndc_x + 1) * vp_w/2 + vp_x
            vp_half_w := to_float(real(to_integer(vp_width))) * to_float(0.5);
            vp_half_h := to_float(real(to_integer(vp_height))) * to_float(0.5);

            sc_x0 := integer(to_real((sx0 + to_float(1.0)) * vp_half_w))
                     + to_integer(vp_x);
            sc_y0 := integer(to_real((to_float(1.0) - sy0) * vp_half_h))
                     + to_integer(vp_y);
            sc_x1 := integer(to_real((sx1 + to_float(1.0)) * vp_half_w))
                     + to_integer(vp_x);
            sc_y1 := integer(to_real((to_float(1.0) - sy1) * vp_half_h))
                     + to_integer(vp_y);
            sc_x2 := integer(to_real((sx2 + to_float(1.0)) * vp_half_w))
                     + to_integer(vp_x);
            sc_y2 := integer(to_real((to_float(1.0) - sy2) * vp_half_h))
                     + to_integer(vp_y);

            -- Edge vectors (integer, 4 bits subpixel precision × 16 = Q4 fixed)
            dx01 <= to_signed(sc_x1 - sc_x0, 32);
            dy01 <= to_signed(sc_y1 - sc_y0, 32);
            dx12 <= to_signed(sc_x2 - sc_x1, 32);
            dy12 <= to_signed(sc_y2 - sc_y1, 32);
            dx20 <= to_signed(sc_x0 - sc_x2, 32);
            dy20 <= to_signed(sc_y0 - sc_y2, 32);

            su_state <= SU_EDGE;

          when SU_EDGE =>
            -- Edge equation coefficients (A*x + B*y + C >= 0 for inside)
            -- Edge 0 (V0→V1):  A = -(dy), B = dx, C = -(A*x0 + B*y0)
            e0_a <= -dy01; e0_b <= dx01;
            e0_c <= -(-dy01 * to_signed(integer(to_real(sx0)*16.0), 32) +
                       dx01 * to_signed(integer(to_real(sy0)*16.0), 32));

            e1_a <= -dy12; e1_b <= dx12;
            e1_c <= -(-dy12 * to_signed(integer(to_real(sx1)*16.0), 32) +
                       dx12 * to_signed(integer(to_real(sy1)*16.0), 32));

            e2_a <= -dy20; e2_b <= dx20;
            e2_c <= -(-dy20 * to_signed(integer(to_real(sx2)*16.0), 32) +
                       dx20 * to_signed(integer(to_real(sy2)*16.0), 32));

            -- Signed area (cross product) × 2
            area_2x <= dx01 * (-dy20) - dy01 * (-dx20);

            -- Back-face: negative area = CW winding → back face
            if (dx01 * (-dy20) - dy01 * (-dx20)) <= 0 then
              backface <= '1';
            else
              backface <= '0';
            end if;

            -- Reciprocal area for barycentric weights
            area_v := to_integer(dx01 * (-dy20) - dy01 * (-dx20));
            if area_v /= 0 then
              inv_a := to_float(1.0) / to_float(real(area_v));
            else
              inv_a := to_float(0.0);
            end if;
            inv_area <= to_slv(inv_a);

            -- Z gradient across triangle
            dzdx_v := (sz1 - sz0) * inv_a;
            dzdy_v := (sz2 - sz0) * inv_a;
            dzdx <= to_slv(dzdx_v);
            dzdy <= to_slv(dzdy_v);

            su_state <= SU_BBOX;

          when SU_BBOX =>
            -- Screen-space bounding box, clamped to viewport
            bx0 := 0; by0 := 0;
            bx1 := to_integer(vp_width)-1;
            by1 := to_integer(vp_height)-1;

            bbox_x0 <= to_unsigned(bx0, 16);
            bbox_y0 <= to_unsigned(by0, 16);
            bbox_x1 <= to_unsigned(bx1, 16);
            bbox_y1 <= to_unsigned(by1, 16);
            su_state <= SU_DONE;

          when SU_DONE =>
            setup_valid <= '1';
            su_state    <= SU_IDLE;
        end case;
      end if;
    end if;
  end process p_setup;

end architecture rtl;


-- =============================================================================
-- Fine Rasteriser + Coverage Mask
-- =============================================================================
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_fine_raster is
  generic (
    TILE_SIZE  : integer := 16;    -- 16×16 pixel tiles
    MSAA_SAMP  : integer := 4      -- 4x MSAA
  );
  port (
    clk        : in  std_logic;
    rst_n      : in  std_logic;

    -- From triangle setup
    setup_valid: in  std_logic;
    e0_a, e0_b, e0_c : in signed(31 downto 0);
    e1_a, e1_b, e1_c : in signed(31 downto 0);
    e2_a, e2_b, e2_c : in signed(31 downto 0);
    bbox_x0, bbox_y0  : in unsigned(15 downto 0);
    bbox_x1, bbox_y1  : in unsigned(15 downto 0);
    inv_area   : in  std_logic_vector(DATA_WIDTH-1 downto 0);
    dzdx, dzdy : in  std_logic_vector(DATA_WIDTH-1 downto 0);

    -- Pixel output (one pixel per cycle)
    pix_valid  : out std_logic;
    pix_x      : out unsigned(15 downto 0);
    pix_y      : out unsigned(15 downto 0);
    pix_cov    : out std_logic_vector(MSAA_SAMP-1 downto 0);  -- Coverage mask
    pix_bary_u : out std_logic_vector(DATA_WIDTH-1 downto 0); -- Barycentric u
    pix_bary_v : out std_logic_vector(DATA_WIDTH-1 downto 0); -- Barycentric v
    pix_depth  : out std_logic_vector(DATA_WIDTH-1 downto 0); -- Interpolated Z
    raster_done: out std_logic
  );
end entity krafgpu_fine_raster;

architecture rtl of krafgpu_fine_raster is

  type rast_state_t is (RS_IDLE, RS_SCAN, RS_PIXEL, RS_DONE);
  signal rs_state  : rast_state_t := RS_IDLE;

  signal cur_x     : unsigned(15 downto 0) := (others => '0');
  signal cur_y     : unsigned(15 downto 0) := (others => '0');

  -- Edge function values at current pixel (integer fixed-point)
  signal ef0, ef1, ef2 : signed(31 downto 0);

  -- MSAA sample offsets (4x: Rotated-grid pattern)
  type samp_offset_t is array (0 to 3) of signed(3 downto 0);
  constant MSAA_OFF_X : samp_offset_t := (to_signed(-3,4), to_signed(1,4),
                                          to_signed(3,4), to_signed(-1,4));
  constant MSAA_OFF_Y : samp_offset_t := (to_signed(-1,4), to_signed(-3,4),
                                          to_signed(1,4), to_signed(3,4));

  signal pix_valid_r : std_logic := '0';
  signal pix_x_r     : unsigned(15 downto 0);
  signal pix_y_r     : unsigned(15 downto 0);
  signal pix_cov_r   : std_logic_vector(MSAA_SAMP-1 downto 0);
  signal pix_depth_r : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal done_r      : std_logic := '0';

begin

  p_rast : process(clk)
    variable ef0v, ef1v, ef2v : signed(31 downto 0);
    variable cov : std_logic_vector(MSAA_SAMP-1 downto 0);
    variable samp_ef0, samp_ef1, samp_ef2 : signed(31 downto 0);
    variable depth_v : signed(31 downto 0);
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        rs_state    <= RS_IDLE;
        pix_valid_r <= '0';
        done_r      <= '0';
      else
        pix_valid_r <= '0';
        done_r      <= '0';

        case rs_state is
          when RS_IDLE =>
            if setup_valid = '1' then
              cur_x    <= bbox_x0;
              cur_y    <= bbox_y0;
              -- Evaluate edge functions at top-left corner.
              -- bbox_x0/y0 are unsigned(15 downto 0); widen to 18 bits via
              -- resize() before the signed reinterpretation feeding the
              -- 32-bit signed multiply below (avoids the ambiguous
              -- "00" & unsigned literal-concatenation idiom).
              ef0 <= e0_a * signed(resize(bbox_x0, 18)) +
                     e0_b * signed(resize(bbox_y0, 18)) + e0_c;
              ef1 <= e1_a * signed(resize(bbox_x0, 18)) +
                     e1_b * signed(resize(bbox_y0, 18)) + e1_c;
              ef2 <= e2_a * signed(resize(bbox_x0, 18)) +
                     e2_b * signed(resize(bbox_y0, 18)) + e2_c;
              rs_state <= RS_SCAN;
            end if;

          when RS_SCAN =>
            -- Check if current pixel is inside triangle
            ef0v := ef0; ef1v := ef1; ef2v := ef2;

            -- MSAA coverage mask: test each sample point
            cov := (others => '0');
            for s in 0 to MSAA_SAMP-1 loop
              samp_ef0 := ef0v + e0_a * MSAA_OFF_X(s) + e0_b * MSAA_OFF_Y(s);
              samp_ef1 := ef1v + e1_a * MSAA_OFF_X(s) + e1_b * MSAA_OFF_Y(s);
              samp_ef2 := ef2v + e2_a * MSAA_OFF_X(s) + e2_b * MSAA_OFF_Y(s);
              if samp_ef0 >= 0 and samp_ef1 >= 0 and samp_ef2 >= 0 then
                cov(s) := '1';
              end if;
            end loop;

            -- If any sample covered, emit pixel
            if cov /= (MSAA_SAMP-1 downto 0 => '0') then
              pix_x_r     <= cur_x;
              pix_y_r     <= cur_y;
              pix_cov_r   <= cov;
              -- Depth at pixel centre: Z0 + dzdx*(x-x0) + dzdy*(y-y0)
              depth_v     := signed(dzdx) * signed(resize(cur_x, 18)) +
                             signed(dzdy) * signed(resize(cur_y, 18));
              pix_depth_r <= std_logic_vector(depth_v(DATA_WIDTH-1 downto 0));
              pix_valid_r <= '1';
              rs_state    <= RS_PIXEL;
            else
              rs_state    <= RS_PIXEL;
            end if;

          when RS_PIXEL =>
            -- Advance to next pixel
            if cur_x < bbox_x1 then
              cur_x <= cur_x + 1;
              -- Increment edge functions by A (step in X)
              ef0 <= ef0 + e0_a;
              ef1 <= ef1 + e1_a;
              ef2 <= ef2 + e2_a;
              rs_state <= RS_SCAN;
            elsif cur_y < bbox_y1 then
              -- Next row: reset X, step Y
              cur_x <= bbox_x0;
              cur_y <= cur_y + 1;
              ef0 <= e0_a * signed(resize(bbox_x0, 18)) +
                     e0_b * signed(resize(cur_y+1, 18)) + e0_c;
              ef1 <= e1_a * signed(resize(bbox_x0, 18)) +
                     e1_b * signed(resize(cur_y+1, 18)) + e1_c;
              ef2 <= e2_a * signed(resize(bbox_x0, 18)) +
                     e2_b * signed(resize(cur_y+1, 18)) + e2_c;
              rs_state <= RS_SCAN;
            else
              rs_state <= RS_DONE;
            end if;

          when RS_DONE =>
            done_r   <= '1';
            rs_state <= RS_IDLE;
        end case;
      end if;
    end if;
  end process p_rast;

  pix_valid  <= pix_valid_r;
  pix_x      <= pix_x_r;
  pix_y      <= pix_y_r;
  pix_cov    <= pix_cov_r;
  pix_depth  <= pix_depth_r;
  pix_bary_u <= (others => '0');  -- Extended in attribute interpolator
  pix_bary_v <= (others => '0');
  raster_done<= done_r;

end architecture rtl;


-- =============================================================================
-- Depth Buffer (Hi-Z + per-pixel Z)
-- =============================================================================
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_depth_buffer is
  generic (
    FB_WIDTH  : integer := 3840;   -- 4K width
    FB_HEIGHT : integer := 2160;   -- 4K height
    HIZ_TILE  : integer := 8       -- Hi-Z tile size (8×8 pixels)
  );
  port (
    clk         : in  std_logic;
    rst_n       : in  std_logic;

    -- Clear depth buffer
    clear_en    : in  std_logic;
    clear_val   : in  std_logic_vector(DATA_WIDTH-1 downto 0);  -- Usually 1.0

    -- Test + update request
    test_valid  : in  std_logic;
    test_x      : in  unsigned(15 downto 0);
    test_y      : in  unsigned(15 downto 0);
    test_depth  : in  std_logic_vector(DATA_WIDTH-1 downto 0);
    test_write  : in  std_logic;   -- 1=write on pass

    -- Test result
    test_pass   : out std_logic;
    test_done   : out std_logic;

    -- Memory interface (depth buffer stored in VRAM)
    mem_req     : out mem_req_t;
    mem_resp    : in  mem_resp_t
  );
end entity krafgpu_depth_buffer;

architecture rtl of krafgpu_depth_buffer is

  -- Hi-Z tile: stores min-Z per 8×8 block for fast rejection
  constant HIZ_TILES_X : integer := FB_WIDTH  / HIZ_TILE;  -- 480
  constant HIZ_TILES_Y : integer := FB_HEIGHT / HIZ_TILE;  -- 270

  type hiz_t is array (0 to HIZ_TILES_X*HIZ_TILES_Y-1) of
                std_logic_vector(DATA_WIDTH-1 downto 0);
  signal hiz_min : hiz_t := (others => (others => '1'));  -- Init to max Z

  type depth_state_t is (DZ_IDLE, DZ_HIZ_TEST, DZ_FETCH, DZ_WAIT,
                         DZ_COMPARE, DZ_WRITE, DZ_DONE);
  signal dz_state   : depth_state_t := DZ_IDLE;

  signal hiz_idx    : integer range 0 to HIZ_TILES_X*HIZ_TILES_Y-1;
  signal fetched_z  : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal pass_r     : std_logic := '0';
  signal done_r     : std_logic := '0';
  signal clear_cnt  : integer range 0 to HIZ_TILES_X*HIZ_TILES_Y-1 := 0;
  signal clearing   : std_logic := '0';

begin

  hiz_idx <= (to_integer(test_y) / HIZ_TILE) * HIZ_TILES_X +
             (to_integer(test_x) / HIZ_TILE);

  p_depth : process(clk)
    variable new_z, old_z : std_logic_vector(DATA_WIDTH-1 downto 0);
    variable pix_addr : unsigned(PHYS_ADDR_WIDTH-1 downto 0);
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        dz_state <= DZ_IDLE;
        pass_r   <= '0';
        done_r   <= '0';
        clearing <= '0';
        mem_req.valid <= '0';

      else
        done_r        <= '0';
        mem_req.valid <= '0';

        -- Asynchronous clear
        if clear_en = '1' then
          clearing  <= '1';
          clear_cnt <= 0;
        end if;

        if clearing = '1' then
          hiz_min(clear_cnt) <= clear_val;
          if clear_cnt = HIZ_TILES_X*HIZ_TILES_Y-1 then
            clearing  <= '0';
          else
            clear_cnt <= clear_cnt + 1;
          end if;
        end if;

        case dz_state is
          when DZ_IDLE =>
            pass_r <= '0';
            if test_valid = '1' and clearing = '0' then
              dz_state <= DZ_HIZ_TEST;
            end if;

          when DZ_HIZ_TEST =>
            -- Hi-Z: if incoming Z >= tile min-Z, reject immediately
            if unsigned(test_depth) >= unsigned(hiz_min(hiz_idx)) then
              pass_r   <= '0';
              done_r   <= '1';
              dz_state <= DZ_IDLE;
            else
              dz_state <= DZ_FETCH;
            end if;

          when DZ_FETCH =>
            -- Fetch per-pixel Z from depth buffer (stored in VRAM)
            pix_addr := to_unsigned(
              to_integer(test_y) * FB_WIDTH + to_integer(test_x),
              PHYS_ADDR_WIDTH) * 4;  -- 4 bytes per pixel
            mem_req.valid    <= '1';
            mem_req.addr     <= pix_addr;
            mem_req.write_en <= '0';
            mem_req.data     <= (others => '0');
            mem_req.byte_en  <= "1111";
            mem_req.warp_id  <= (others => '0');
            mem_req.req_id   <= (others => '0');
            dz_state         <= DZ_WAIT;

          when DZ_WAIT =>
            if mem_resp.valid = '1' then
              fetched_z <= mem_resp.data;
              dz_state  <= DZ_COMPARE;
            end if;

          when DZ_COMPARE =>
            -- LESS comparison: pass if incoming_Z < stored_Z
            if unsigned(test_depth) < unsigned(fetched_z) then
              pass_r <= '1';
              if test_write = '1' then
                dz_state <= DZ_WRITE;
              else
                done_r   <= '1';
                dz_state <= DZ_IDLE;
              end if;
            else
              pass_r   <= '0';
              done_r   <= '1';
              dz_state <= DZ_IDLE;
            end if;

          when DZ_WRITE =>
            -- Write new Z to depth buffer
            pix_addr := to_unsigned(
              to_integer(test_y) * FB_WIDTH + to_integer(test_x),
              PHYS_ADDR_WIDTH) * 4;
            mem_req.valid    <= '1';
            mem_req.addr     <= pix_addr;
            mem_req.write_en <= '1';
            mem_req.data     <= test_depth;
            mem_req.byte_en  <= "1111";
            mem_req.warp_id  <= (others => '0');
            mem_req.req_id   <= to_unsigned(1, 8);

            -- Update Hi-Z tile minimum
            if unsigned(test_depth) < unsigned(hiz_min(hiz_idx)) then
              hiz_min(hiz_idx) <= test_depth;
            end if;
            dz_state <= DZ_DONE;

          when DZ_DONE =>
            done_r   <= '1';
            dz_state <= DZ_IDLE;
        end case;
      end if;
    end if;
  end process p_depth;

  test_pass <= pass_r;
  test_done <= done_r;

end architecture rtl;
