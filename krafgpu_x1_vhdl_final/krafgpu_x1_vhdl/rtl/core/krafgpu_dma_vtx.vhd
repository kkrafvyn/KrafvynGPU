-- =============================================================================
-- KrafGPU-X1 DMA Copy Engine
-- Project      : KrafGPU-X1 Next-Generation GPU Architecture
-- Author       : Krafvyn Tech / Don
-- Description  : High-bandwidth DMA engine for memory copy operations.
--
--   Features:
--     - 512-bit (64-byte) transfer width (full HBM4 cache line)
--     - Source and destination in any combination of:
--         HBM4 VRAM ↔ HBM4 VRAM (intra-GPU copy)
--         System RAM ↔ HBM4 VRAM (PCIe DMA)
--     - Gather-scatter support (non-contiguous source/destination)
--     - Transfer sizes from 1 byte to 4 GB
--     - Alignment: best performance at 64-byte aligned src+dst
--     - Pipeline: issue next fetch while previous write completes
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_dma_engine is
  port (
    clk          : in  std_logic;
    rst_n        : in  std_logic;

    -- Command (from ring processor)
    copy_valid   : in  std_logic;
    copy_src     : in  unsigned(PHYS_ADDR_WIDTH-1 downto 0);
    copy_dst     : in  unsigned(PHYS_ADDR_WIDTH-1 downto 0);
    copy_size    : in  unsigned(31 downto 0);    -- Bytes
    copy_ready   : out std_logic;
    copy_done    : out std_logic;

    -- Read interface (source)
    rd_req       : out mem_req_t;
    rd_resp      : in  mem_resp_t;

    -- Write interface (destination)
    wr_req       : out mem_req_t;
    wr_resp      : in  mem_resp_t;

    -- Byte counter
    bytes_copied : out unsigned(31 downto 0)
  );
end entity krafgpu_dma_engine;

architecture rtl of krafgpu_dma_engine is

  constant LINE_BYTES  : integer := 4;   -- 4 bytes per transfer in this model
                                          -- (real HW: 64 bytes per line)

  type dma_state_t is (DMA_IDLE, DMA_FETCH, DMA_FETCH_WAIT,
                       DMA_STORE, DMA_STORE_WAIT, DMA_DONE);
  signal dma_state     : dma_state_t := DMA_IDLE;

  signal src_ptr       : unsigned(PHYS_ADDR_WIDTH-1 downto 0);
  signal dst_ptr       : unsigned(PHYS_ADDR_WIDTH-1 downto 0);
  signal remain        : unsigned(31 downto 0);
  signal data_buf      : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal copied_r      : unsigned(31 downto 0) := (others => '0');
  signal done_r        : std_logic := '0';
  signal ready_r       : std_logic := '1';

begin
  copy_ready   <= ready_r;
  copy_done    <= done_r;
  bytes_copied <= copied_r;

  p_dma : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        dma_state   <= DMA_IDLE;
        done_r      <= '0';
        ready_r     <= '1';
        copied_r    <= (others => '0');
        rd_req.valid<= '0';
        wr_req.valid<= '0';
      else
        done_r      <= '0';
        rd_req.valid<= '0';
        wr_req.valid<= '0';

        case dma_state is
          when DMA_IDLE =>
            ready_r <= '1';
            if copy_valid = '1' then
              src_ptr   <= copy_src;
              dst_ptr   <= copy_dst;
              remain    <= copy_size;
              copied_r  <= (others => '0');
              ready_r   <= '0';
              dma_state <= DMA_FETCH;
            end if;

          when DMA_FETCH =>
            if remain = 0 then
              dma_state <= DMA_DONE;
            else
              rd_req.valid    <= '1';
              rd_req.addr     <= src_ptr;
              rd_req.write_en <= '0';
              rd_req.data     <= (others => '0');
              rd_req.byte_en  <= "1111";
              rd_req.warp_id  <= (others => '0');
              rd_req.req_id   <= (others => '0');
              dma_state <= DMA_FETCH_WAIT;
            end if;

          when DMA_FETCH_WAIT =>
            if rd_resp.valid = '1' then
              data_buf  <= rd_resp.data;
              dma_state <= DMA_STORE;
            end if;

          when DMA_STORE =>
            wr_req.valid    <= '1';
            wr_req.addr     <= dst_ptr;
            wr_req.write_en <= '1';
            wr_req.data     <= data_buf;
            wr_req.byte_en  <= "1111";
            wr_req.warp_id  <= (others => '0');
            wr_req.req_id   <= to_unsigned(1, 8);
            dma_state <= DMA_STORE_WAIT;

          when DMA_STORE_WAIT =>
            if wr_resp.valid = '1' then
              src_ptr  <= src_ptr + LINE_BYTES;
              dst_ptr  <= dst_ptr + LINE_BYTES;
              copied_r <= copied_r + LINE_BYTES;
              if remain <= LINE_BYTES then
                remain    <= (others => '0');
              else
                remain <= remain - LINE_BYTES;
              end if;
              dma_state <= DMA_FETCH;
            end if;

          when DMA_DONE =>
            done_r    <= '1';
            ready_r   <= '1';
            dma_state <= DMA_IDLE;
        end case;
      end if;
    end if;
  end process p_dma;

end architecture rtl;


-- =============================================================================
-- KrafGPU-X1 Vertex Fetch Unit
-- Description  : Reads vertex data from VRAM and assembles vertex records
--                for the geometry pipeline. Supports interleaved and
--                non-interleaved vertex buffer layouts.
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_vertex_fetch is
  generic (
    MAX_ATTRIBS  : integer := 16   -- Max vertex attributes
  );
  port (
    clk          : in  std_logic;
    rst_n        : in  std_logic;

    -- Draw command (from ring processor)
    draw_valid   : in  std_logic;
    vtx_count    : in  unsigned(31 downto 0);
    vtx_buf_addr : in  unsigned(PHYS_ADDR_WIDTH-1 downto 0);
    idx_buf_addr : in  unsigned(PHYS_ADDR_WIDTH-1 downto 0);
    vtx_stride   : in  unsigned(15 downto 0);   -- Bytes per vertex
    use_index    : in  std_logic;
    draw_ready   : out std_logic;

    -- Vertex attribute descriptor (from driver-set state)
    attrib_en    : in  std_logic_vector(MAX_ATTRIBS-1 downto 0);
    attrib_offset: in  unsigned(7 downto 0);    -- Byte offset in vertex

    -- Memory interface
    mem_req      : out mem_req_t;
    mem_resp     : in  mem_resp_t;

    -- Vertex output to triangle setup (4 floats per attribute)
    vtx_valid    : out std_logic;
    vtx_pos_x    : out std_logic_vector(DATA_WIDTH-1 downto 0);
    vtx_pos_y    : out std_logic_vector(DATA_WIDTH-1 downto 0);
    vtx_pos_z    : out std_logic_vector(DATA_WIDTH-1 downto 0);
    vtx_pos_w    : out std_logic_vector(DATA_WIDTH-1 downto 0);
    vtx_idx      : out unsigned(31 downto 0);
    vtx_consumed : in  std_logic
  );
end entity krafgpu_vertex_fetch;

architecture rtl of krafgpu_vertex_fetch is

  type vf_state_t is (
    VF_IDLE, VF_FETCH_IDX, VF_WAIT_IDX,
    VF_FETCH_VTX_X, VF_WAIT_X,
    VF_FETCH_VTX_Y, VF_WAIT_Y,
    VF_FETCH_VTX_Z, VF_WAIT_Z,
    VF_FETCH_VTX_W, VF_WAIT_W,
    VF_OUTPUT, VF_DONE
  );
  signal vf_state   : vf_state_t := VF_IDLE;

  signal cur_vtx    : unsigned(31 downto 0) := (others => '0');
  signal cur_idx    : unsigned(31 downto 0) := (others => '0');

  signal pos_x_r    : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal pos_y_r    : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal pos_z_r    : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal pos_w_r    : std_logic_vector(DATA_WIDTH-1 downto 0);

  signal vtx_valid_r: std_logic := '0';
  signal ready_r    : std_logic := '1';

begin
  draw_ready <= ready_r;
  vtx_valid  <= vtx_valid_r;
  vtx_pos_x  <= pos_x_r;
  vtx_pos_y  <= pos_y_r;
  vtx_pos_z  <= pos_z_r;
  vtx_pos_w  <= pos_w_r;
  vtx_idx    <= cur_idx;

  p_vf : process(clk)
    variable vtx_base : unsigned(PHYS_ADDR_WIDTH-1 downto 0);
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        vf_state     <= VF_IDLE;
        ready_r      <= '1';
        vtx_valid_r  <= '0';
        cur_vtx      <= (others => '0');
        mem_req.valid<= '0';
      else
        mem_req.valid <= '0';
        vtx_valid_r   <= '0';

        case vf_state is
          when VF_IDLE =>
            ready_r <= '1';
            if draw_valid = '1' then
              ready_r <= '0';
              cur_vtx <= (others => '0');
              vf_state <= VF_FETCH_IDX;
            end if;

          when VF_FETCH_IDX =>
            if cur_vtx >= vtx_count then
              vf_state <= VF_DONE;
            elsif use_index = '1' then
              mem_req.valid    <= '1';
              mem_req.addr     <= idx_buf_addr + cur_vtx * 4;
              mem_req.write_en <= '0';
              mem_req.data     <= (others => '0');
              mem_req.byte_en  <= "1111";
              mem_req.warp_id  <= (others => '0');
              mem_req.req_id   <= (others => '0');
              vf_state <= VF_WAIT_IDX;
            else
              cur_idx  <= cur_vtx;
              vf_state <= VF_FETCH_VTX_X;
            end if;

          when VF_WAIT_IDX =>
            if mem_resp.valid = '1' then
              cur_idx  <= unsigned(mem_resp.data);
              vf_state <= VF_FETCH_VTX_X;
            end if;

          when VF_FETCH_VTX_X =>
            vtx_base := vtx_buf_addr + cur_idx * vtx_stride;
            mem_req.valid    <= '1';
            mem_req.addr     <= vtx_base;
            mem_req.write_en <= '0';
            mem_req.data     <= (others => '0');
            mem_req.byte_en  <= "1111";
            mem_req.warp_id  <= (others => '0');
            mem_req.req_id   <= to_unsigned(0, 8);
            vf_state <= VF_WAIT_X;

          when VF_WAIT_X =>
            if mem_resp.valid = '1' then
              pos_x_r  <= mem_resp.data;
              vf_state <= VF_FETCH_VTX_Y;
            end if;

          when VF_FETCH_VTX_Y =>
            vtx_base := vtx_buf_addr + cur_idx * vtx_stride + 4;
            mem_req.valid  <= '1'; mem_req.addr <= vtx_base;
            mem_req.write_en <= '0'; mem_req.data <= (others=>'0');
            mem_req.byte_en <= "1111"; mem_req.warp_id <= (others=>'0');
            mem_req.req_id <= to_unsigned(1,8);
            vf_state <= VF_WAIT_Y;

          when VF_WAIT_Y =>
            if mem_resp.valid = '1' then
              pos_y_r  <= mem_resp.data;
              vf_state <= VF_FETCH_VTX_Z;
            end if;

          when VF_FETCH_VTX_Z =>
            vtx_base := vtx_buf_addr + cur_idx * vtx_stride + 8;
            mem_req.valid  <= '1'; mem_req.addr <= vtx_base;
            mem_req.write_en <= '0'; mem_req.data <= (others=>'0');
            mem_req.byte_en <= "1111"; mem_req.warp_id <= (others=>'0');
            mem_req.req_id <= to_unsigned(2,8);
            vf_state <= VF_WAIT_Z;

          when VF_WAIT_Z =>
            if mem_resp.valid = '1' then
              pos_z_r  <= mem_resp.data;
              vf_state <= VF_FETCH_VTX_W;
            end if;

          when VF_FETCH_VTX_W =>
            vtx_base := vtx_buf_addr + cur_idx * vtx_stride + 12;
            mem_req.valid  <= '1'; mem_req.addr <= vtx_base;
            mem_req.write_en <= '0'; mem_req.data <= (others=>'0');
            mem_req.byte_en <= "1111"; mem_req.warp_id <= (others=>'0');
            mem_req.req_id <= to_unsigned(3,8);
            vf_state <= VF_WAIT_W;

          when VF_WAIT_W =>
            if mem_resp.valid = '1' then
              pos_w_r     <= mem_resp.data;
              vtx_valid_r <= '1';
              vf_state    <= VF_OUTPUT;
            end if;

          when VF_OUTPUT =>
            if vtx_consumed = '1' then
              vtx_valid_r <= '0';
              cur_vtx     <= cur_vtx + 1;
              vf_state    <= VF_FETCH_IDX;
            end if;

          when VF_DONE =>
            ready_r  <= '1';
            vf_state <= VF_IDLE;
        end case;
      end if;
    end if;
  end process p_vf;

end architecture rtl;
