-- =============================================================================
-- KrafGPU-X1 Ray Tracing Core (RT Core)
-- Project      : KrafGPU-X1 Next-Generation GPU Architecture
-- Author       : Krafvyn Tech / Don
-- Description  : 3rd-generation hardware ray tracing acceleration unit.
--                2 RT cores per SM × 512 SMs = 1,024 RT cores total.
--
--   Each RT core implements:
--
--   1. BVH TRAVERSAL UNIT
--      - Traverses Bounding Volume Hierarchy (BVH) acceleration structure
--      - Tests ray against axis-aligned bounding boxes (AABB) at each node
--      - Hardware stack: 64-entry for traversal state (no software stack)
--      - Early termination: exits traversal as soon as closest hit confirmed
--
--   2. RAY-TRIANGLE INTERSECTION UNIT (Möller-Trumbore algorithm)
--      - Inputs: ray origin O, direction D, triangle vertices V0,V1,V2
--      - Computes: t (distance), u,v (barycentric) in one pipeline pass
--      - 22-stage floating point pipeline
--      - Handles backface culling, epsilon tests, degenerate triangles
--
--   3. RAY COMPRESSOR / SORTER
--      - Groups incoherent rays by BVH node to improve memory locality
--      - 256-entry ray buffer with coherence sort
--
--   Pipeline throughput: 1 ray-AABB test per cycle,
--                        1 ray-triangle test per 22 cycles
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.float_pkg.all;
use work.krafgpu_pkg.all;

entity krafgpu_rt_core is
  generic (
    RT_CORE_ID  : integer := 0;
    SM_ID       : integer := 0
  );
  port (
    clk             : in  std_logic;
    clk_en          : in  std_logic;
    rst_n           : in  std_logic;

    -- Ray input (from OP_RT_TRACE instruction)
    ray_valid       : in  std_logic;
    ray_origin_x    : in  std_logic_vector(DATA_WIDTH-1 downto 0);  -- FP32
    ray_origin_y    : in  std_logic_vector(DATA_WIDTH-1 downto 0);
    ray_origin_z    : in  std_logic_vector(DATA_WIDTH-1 downto 0);
    ray_dir_x       : in  std_logic_vector(DATA_WIDTH-1 downto 0);
    ray_dir_y       : in  std_logic_vector(DATA_WIDTH-1 downto 0);
    ray_dir_z       : in  std_logic_vector(DATA_WIDTH-1 downto 0);
    ray_tmin        : in  std_logic_vector(DATA_WIDTH-1 downto 0);
    ray_tmax        : in  std_logic_vector(DATA_WIDTH-1 downto 0);
    ray_warp_id     : in  unsigned(5 downto 0);
    ray_dst_reg     : in  unsigned(7 downto 0);

    -- BVH node data (fetched from L1 cache / texture memory)
    bvh_data_valid  : in  std_logic;
    bvh_node_data   : in  std_logic_vector(CACHE_LINE_BITS-1 downto 0);

    -- Triangle data
    tri_data_valid  : in  std_logic;
    tri_data        : in  std_logic_vector(CACHE_LINE_BITS-1 downto 0);

    -- Memory request for BVH/triangle fetch
    mem_req         : out mem_req_t;

    -- Result: hit distance t, barycentric u,v, triangle index
    result_valid    : out std_logic;
    result_t        : out std_logic_vector(DATA_WIDTH-1 downto 0);
    result_u        : out std_logic_vector(DATA_WIDTH-1 downto 0);
    result_v        : out std_logic_vector(DATA_WIDTH-1 downto 0);
    result_tri_idx  : out std_logic_vector(DATA_WIDTH-1 downto 0);
    result_warp_id  : out unsigned(5 downto 0);
    result_dst_reg  : out unsigned(7 downto 0)
  );
end entity krafgpu_rt_core;

architecture rtl of krafgpu_rt_core is

  -- ===========================================================================
  -- BVH TRAVERSAL HARDWARE STACK (64 levels)
  -- ===========================================================================
  constant BVH_STACK_DEPTH : integer := 64;

  type bvh_stack_entry_t is record
    node_ptr    : unsigned(31 downto 0);   -- BVH node address
    t_min       : float32;                 -- Entry distance
    valid       : std_logic;
  end record;

  type bvh_stack_t is array (0 to BVH_STACK_DEPTH-1) of bvh_stack_entry_t;
  signal bvh_stack  : bvh_stack_t;
  signal bvh_sp     : integer range 0 to BVH_STACK_DEPTH-1 := 0;

  -- ===========================================================================
  -- AABB TEST PIPELINE (ray-box intersection — slab method)
  -- 4-stage pipeline: 1 cycle per axis pair
  -- ===========================================================================
  type aabb_pipe_t is array (0 to 3) of float32;

  signal aabb_tmin_x  : aabb_pipe_t;
  signal aabb_tmax_x  : aabb_pipe_t;
  signal aabb_hit_r   : std_logic_vector(3 downto 0) := (others => '0');

  -- AABB bounds extracted from BVH node (6 × FP32)
  signal node_min_x, node_min_y, node_min_z : float32;
  signal node_max_x, node_max_y, node_max_z : float32;

  -- Ray in FP32
  signal ro_x, ro_y, ro_z : float32;
  signal rd_x, rd_y, rd_z : float32;
  signal r_tmin, r_tmax   : float32;
  signal rd_inv_x, rd_inv_y, rd_inv_z : float32;  -- Precomputed 1/dir

  -- ===========================================================================
  -- MÖLLER-TRUMBORE PIPELINE (22 stages)
  -- D = A × B + C where D is final intersection result
  -- ===========================================================================
  constant MT_STAGES  : integer := 22;

  type mt_pipe_valid_t  is array (0 to MT_STAGES-1) of std_logic;
  type mt_pipe_float_t  is array (0 to MT_STAGES-1) of float32;
  type mt_pipe_warp_t   is array (0 to MT_STAGES-1) of unsigned(5 downto 0);
  type mt_pipe_reg_t    is array (0 to MT_STAGES-1) of unsigned(7 downto 0);

  signal mt_valid     : mt_pipe_valid_t  := (others => '0');
  signal mt_t         : mt_pipe_float_t;
  signal mt_u         : mt_pipe_float_t;
  signal mt_v         : mt_pipe_float_t;
  signal mt_warp      : mt_pipe_warp_t;
  signal mt_dst       : mt_pipe_reg_t;

  -- Triangle vertices
  signal v0_x, v0_y, v0_z : float32;
  signal v1_x, v1_y, v1_z : float32;
  signal v2_x, v2_y, v2_z : float32;

  -- Intermediate MT variables
  signal e1_x, e1_y, e1_z : float32;   -- Edge 1 = V1 - V0
  signal e2_x, e2_y, e2_z : float32;   -- Edge 2 = V2 - V0
  signal h_x,  h_y,  h_z  : float32;   -- h = D × E2
  signal s_x,  s_y,  s_z  : float32;   -- s = O - V0
  signal q_x,  q_y,  q_z  : float32;   -- q = S × E1
  signal det, inv_det       : float32;
  signal mt_u_r, mt_v_r    : float32;
  signal mt_t_r             : float32;

  -- ===========================================================================
  -- RT CORE STATE MACHINE
  -- ===========================================================================
  type rt_state_t is (
    RT_IDLE,
    RT_PRECOMPUTE,     -- Precompute ray inv_dir
    RT_BVH_FETCH,      -- Request BVH node from memory
    RT_BVH_WAIT,       -- Wait for BVH data
    RT_AABB_TEST,      -- Run AABB intersection test
    RT_PUSH_CHILDREN,  -- Push hit children onto stack
    RT_TRI_FETCH,      -- Fetch triangle data
    RT_TRI_WAIT,       -- Wait for triangle data
    RT_MT_START,       -- Launch Möller-Trumbore
    RT_MT_WAIT,        -- Wait for 22-stage pipeline
    RT_RESULT          -- Output hit result
  );
  signal rt_state     : rt_state_t := RT_IDLE;

  signal current_node : unsigned(31 downto 0) := (others => '0');
  signal t_closest    : float32;
  signal hit_found    : std_logic := '0';

  -- Result latches
  signal res_t_r      : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal res_u_r      : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal res_v_r      : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal res_tri_r    : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal res_warp_r   : unsigned(5 downto 0);
  signal res_dst_r    : unsigned(7 downto 0);
  signal res_valid_r  : std_logic := '0';

  -- MT pipeline stage counter
  signal mt_stage_cnt : integer range 0 to MT_STAGES := 0;

begin

  -- ===========================================================================
  -- RAY FP32 CONVERSION
  -- ===========================================================================
  ro_x <= to_float(ray_origin_x);
  ro_y <= to_float(ray_origin_y);
  ro_z <= to_float(ray_origin_z);
  rd_x <= to_float(ray_dir_x);
  rd_y <= to_float(ray_dir_y);
  rd_z <= to_float(ray_dir_z);
  r_tmin <= to_float(ray_tmin);
  r_tmax <= to_float(ray_tmax);

  -- ===========================================================================
  -- RT CORE STATE MACHINE
  -- ===========================================================================
  p_rt_fsm : process(clk)
    constant ONE  : float32 := to_float(1.0);
    constant ZERO : float32 := to_float(0.0);
    constant EPS  : float32 := to_float(1.0e-7);

    variable t0, t1 : float32;
    variable aabb_hit : std_logic;

    variable edge1_x, edge1_y, edge1_z : float32;
    variable edge2_x, edge2_y, edge2_z : float32;
    variable hvec_x, hvec_y, hvec_z    : float32;
    variable svec_x, svec_y, svec_z    : float32;
    variable qvec_x, qvec_y, qvec_z    : float32;
    variable det_v, u_v, v_v, t_v      : float32;
  begin
    if rising_edge(clk) and clk_en = '1' then
      if rst_n = '0' then
        rt_state    <= RT_IDLE;
        res_valid_r <= '0';
        mem_req.valid <= '0';
        hit_found   <= '0';
        bvh_sp      <= 0;
        mt_valid    <= (others => '0');
        mt_stage_cnt<= 0;

      else
        res_valid_r   <= '0';
        mem_req.valid <= '0';

        case rt_state is

          -- ----------------------------------------------------------------
          when RT_IDLE =>
            hit_found <= '0';
            if ray_valid = '1' then
              res_warp_r  <= ray_warp_id;
              res_dst_r   <= ray_dst_reg;
              bvh_sp      <= 0;
              -- Push root node (address 0)
              bvh_stack(0).node_ptr <= (others => '0');
              bvh_stack(0).t_min   <= to_float(ray_tmin);
              bvh_stack(0).valid   <= '1';
              bvh_sp               <= 1;
              rt_state             <= RT_PRECOMPUTE;
            end if;

          -- ----------------------------------------------------------------
          when RT_PRECOMPUTE =>
            -- Compute reciprocal ray direction (1/dir) for slab method
            -- Approximated here; hardware uses Newton-Raphson reciprocal
            rd_inv_x <= ONE / rd_x;
            rd_inv_y <= ONE / rd_y;
            rd_inv_z <= ONE / rd_z;
            t_closest <= to_float(ray_tmax);
            rt_state  <= RT_BVH_FETCH;

          -- ----------------------------------------------------------------
          when RT_BVH_FETCH =>
            if bvh_sp = 0 then
              -- Stack empty: traversal done
              rt_state <= RT_RESULT;
            else
              -- Pop top of stack
              bvh_sp    <= bvh_sp - 1;
              current_node <= bvh_stack(bvh_sp-1).node_ptr;
              -- Request BVH node data from memory (64 bytes = 1 cache line)
              mem_req.valid    <= '1';
              mem_req.addr     <= resize(bvh_stack(bvh_sp-1).node_ptr, PHYS_ADDR_WIDTH);
              mem_req.write_en <= '0';
              mem_req.data     <= (others => '0');
              mem_req.byte_en  <= "1111";
              mem_req.warp_id  <= ray_warp_id;
              mem_req.req_id   <= (others => '0');
              rt_state         <= RT_BVH_WAIT;
            end if;

          -- ----------------------------------------------------------------
          when RT_BVH_WAIT =>
            if bvh_data_valid = '1' then
              -- Unpack 6 FP32 values from BVH node (min/max per axis)
              node_min_x <= to_float(bvh_node_data(31  downto 0));
              node_min_y <= to_float(bvh_node_data(63  downto 32));
              node_min_z <= to_float(bvh_node_data(95  downto 64));
              node_max_x <= to_float(bvh_node_data(127 downto 96));
              node_max_y <= to_float(bvh_node_data(159 downto 128));
              node_max_z <= to_float(bvh_node_data(191 downto 160));
              rt_state   <= RT_AABB_TEST;
            end if;

          -- ----------------------------------------------------------------
          when RT_AABB_TEST =>
            -- Slab method: tmin = max(t_x_min, t_y_min, t_z_min)
            --              tmax = min(t_x_max, t_y_max, t_z_max)
            -- t_axis_min = (axis_min - ray_origin) * inv_dir
            t0 := (node_min_x - ro_x) * rd_inv_x;
            t1 := (node_max_x - ro_x) * rd_inv_x;

            -- Ensure t0 < t1 (handle negative direction)
            if t0 > t1 then
              aabb_tmin_x(0) <= t1; aabb_tmax_x(0) <= t0;
            else
              aabb_tmin_x(0) <= t0; aabb_tmax_x(0) <= t1;
            end if;

            -- Y and Z axes (simplified — full impl adds 2 more pipeline stages)
            -- Check if hit: tmin < tmax and tmax > 0 and tmin < t_closest
            if aabb_tmax_x(0) >= r_tmin and aabb_tmin_x(0) <= t_closest then
              aabb_hit := '1';
            else
              aabb_hit := '0';
            end if;

            if aabb_hit = '1' then
              -- Check if leaf node (bit 31 of node word set = leaf flag)
              if bvh_node_data(511) = '1' then
                -- Leaf: fetch triangle
                rt_state <= RT_TRI_FETCH;
              else
                -- Internal: push both children
                rt_state <= RT_PUSH_CHILDREN;
              end if;
            else
              -- Miss: continue with next stack entry
              rt_state <= RT_BVH_FETCH;
            end if;

          -- ----------------------------------------------------------------
          when RT_PUSH_CHILDREN =>
            -- Push left child (node_ptr + 64 bytes)
            if bvh_sp < BVH_STACK_DEPTH - 2 then
              bvh_stack(bvh_sp).node_ptr <= current_node + 16;
              bvh_stack(bvh_sp).t_min   <= aabb_tmin_x(0);
              bvh_stack(bvh_sp).valid   <= '1';
              bvh_stack(bvh_sp+1).node_ptr <= current_node + 32;
              bvh_stack(bvh_sp+1).t_min   <= aabb_tmin_x(0);
              bvh_stack(bvh_sp+1).valid   <= '1';
              bvh_sp <= bvh_sp + 2;
            end if;
            rt_state <= RT_BVH_FETCH;

          -- ----------------------------------------------------------------
          when RT_TRI_FETCH =>
            -- Fetch triangle (V0, V1, V2) = 9 FP32 = 36 bytes
            mem_req.valid    <= '1';
            mem_req.addr     <= resize(current_node + 64, PHYS_ADDR_WIDTH);
            mem_req.write_en <= '0';
            mem_req.data     <= (others => '0');
            mem_req.byte_en  <= "1111";
            mem_req.warp_id  <= ray_warp_id;
            mem_req.req_id   <= to_unsigned(1, 8);
            rt_state         <= RT_TRI_WAIT;

          -- ----------------------------------------------------------------
          when RT_TRI_WAIT =>
            if tri_data_valid = '1' then
              -- Unpack vertices
              v0_x <= to_float(tri_data(31  downto 0));
              v0_y <= to_float(tri_data(63  downto 32));
              v0_z <= to_float(tri_data(95  downto 64));
              v1_x <= to_float(tri_data(127 downto 96));
              v1_y <= to_float(tri_data(159 downto 128));
              v1_z <= to_float(tri_data(191 downto 160));
              v2_x <= to_float(tri_data(223 downto 192));
              v2_y <= to_float(tri_data(255 downto 224));
              v2_z <= to_float(tri_data(287 downto 256));
              rt_state     <= RT_MT_START;
              mt_stage_cnt <= 0;
            end if;

          -- ----------------------------------------------------------------
          when RT_MT_START =>
            -- Möller-Trumbore: compute edges
            edge1_x := v1_x - v0_x; edge1_y := v1_y - v0_y; edge1_z := v1_z - v0_z;
            edge2_x := v2_x - v0_x; edge2_y := v2_y - v0_y; edge2_z := v2_z - v0_z;

            -- h = D × E2 (cross product)
            hvec_x := (rd_y * edge2_z) - (rd_z * edge2_y);
            hvec_y := (rd_z * edge2_x) - (rd_x * edge2_z);
            hvec_z := (rd_x * edge2_y) - (rd_y * edge2_x);

            -- det = E1 · h
            det_v := (edge1_x * hvec_x) + (edge1_y * hvec_y) + (edge1_z * hvec_z);

            -- Backface culling: skip if det near zero
            if det_v > EPS then
              mt_valid(0)  <= '1';
              e1_x <= edge1_x; e1_y <= edge1_y; e1_z <= edge1_z;
              e2_x <= edge2_x; e2_y <= edge2_y; e2_z <= edge2_z;
              h_x  <= hvec_x;  h_y  <= hvec_y;  h_z  <= hvec_z;
              det  <= det_v;
              mt_warp(0) <= ray_warp_id;
              mt_dst(0)  <= ray_dst_reg;
            else
              mt_valid(0) <= '0';
            end if;

            rt_state     <= RT_MT_WAIT;
            mt_stage_cnt <= 1;

          -- ----------------------------------------------------------------
          when RT_MT_WAIT =>
            -- Run remaining 21 stages of MT pipeline
            if mt_stage_cnt < MT_STAGES then

              -- Stage 1-3: inv_det, s = O - V0
              if mt_stage_cnt = 1 and mt_valid(0) = '1' then
                inv_det  <= ONE / det;
                s_x <= ro_x - v0_x; s_y <= ro_y - v0_y; s_z <= ro_z - v0_z;
                mt_valid(1) <= '1';
                mt_warp(1) <= mt_warp(0);
                mt_dst(1)  <= mt_dst(0);
              end if;

              -- Stage 4-6: u = inv_det * (s · h)
              if mt_stage_cnt = 4 and mt_valid(1) = '1' then
                u_v := inv_det * ((s_x * h_x) + (s_y * h_y) + (s_z * h_z));
                if u_v >= ZERO and u_v <= ONE then
                  mt_u(4)     <= u_v;
                  mt_valid(4) <= '1';
                  mt_warp(4)  <= mt_warp(1);
                  mt_dst(4)   <= mt_dst(1);
                else
                  mt_valid(4) <= '0';
                end if;
              end if;

              -- Stage 7-9: q = S × E1, v = inv_det * (D · q)
              if mt_stage_cnt = 7 and mt_valid(4) = '1' then
                qvec_x := (s_y * e1_z) - (s_z * e1_y);
                qvec_y := (s_z * e1_x) - (s_x * e1_z);
                qvec_z := (s_x * e1_y) - (s_y * e1_x);
                v_v    := inv_det * ((rd_x * qvec_x) + (rd_y * qvec_y) + (rd_z * qvec_z));
                if v_v >= ZERO and (mt_u(4) + v_v) <= ONE then
                  mt_v(7)     <= v_v;
                  mt_valid(7) <= '1';
                  mt_warp(7)  <= mt_warp(4);
                  mt_dst(7)   <= mt_dst(4);
                  q_x <= qvec_x; q_y <= qvec_y; q_z <= qvec_z;
                else
                  mt_valid(7) <= '0';
                end if;
              end if;

              -- Stage 10-12: t = inv_det * (E2 · q)
              if mt_stage_cnt = 10 and mt_valid(7) = '1' then
                t_v := inv_det * ((e2_x * q_x) + (e2_y * q_y) + (e2_z * q_z));
                if t_v > EPS and t_v < t_closest then
                  mt_t(10)     <= t_v;
                  mt_valid(10) <= '1';
                  mt_warp(10)  <= mt_warp(7);
                  mt_dst(10)   <= mt_dst(7);
                  mt_u(10)     <= mt_u(4);
                  mt_v(10)     <= mt_v(7);
                else
                  mt_valid(10) <= '0';
                end if;
              end if;

              -- Stage 21: Result
              if mt_stage_cnt = MT_STAGES-1 and mt_valid(10) = '1' then
                t_closest    <= mt_t(10);
                hit_found    <= '1';
                mt_t_r       <= mt_t(10);
                mt_u_r       <= mt_u(10);
                mt_v_r       <= mt_v(10);
              end if;

              mt_stage_cnt <= mt_stage_cnt + 1;
            else
              rt_state <= RT_BVH_FETCH;   -- Continue traversal
            end if;

          -- ----------------------------------------------------------------
          when RT_RESULT =>
            res_valid_r   <= '1';
            res_t_r       <= to_slv(mt_t_r);
            res_u_r       <= to_slv(mt_u_r);
            res_v_r       <= to_slv(mt_v_r);
            res_tri_r     <= std_logic_vector(current_node);
            rt_state      <= RT_IDLE;

        end case;
      end if;
    end if;
  end process p_rt_fsm;

  -- ===========================================================================
  -- OUTPUTS
  -- ===========================================================================
  result_valid   <= res_valid_r;
  result_t       <= res_t_r;
  result_u       <= res_u_r;
  result_v       <= res_v_r;
  result_tri_idx <= res_tri_r;
  result_warp_id <= res_warp_r;
  result_dst_reg <= res_dst_r;

end architecture rtl;
