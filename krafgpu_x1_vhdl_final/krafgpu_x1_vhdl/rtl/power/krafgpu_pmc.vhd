-- =============================================================================
-- KrafGPU-X1 Power Management Controller (PMC)
-- Project      : KrafGPU-X1 Next-Generation GPU Architecture
-- Author       : Krafvyn Tech / Don
-- Description  : ARM Cortex-M33 equivalent PMC implemented in VHDL.
--                Runs a 1ms control loop:
--                  1. Samples telemetry from all 16 voltage islands
--                  2. Predicts next workload demand (EMA filter)
--                  3. Decides V/F state per island
--                  4. Issues PMBus VOUT_COMMAND to 4× PMIC
--                  5. Guards thermal limits (TJ < 95°C)
--                  6. Logs events over UART
--                Clocked at 400 MHz (always-on domain, never power-gated).
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.krafgpu_pkg.all;

entity krafgpu_pmc is
  port (
    clk               : in  std_logic;   -- 400 MHz always-on
    rst_n             : in  std_logic;

    -- Telemetry inputs from both chiplets (16 islands total)
    telemetry_in      : in  telemetry_array_t;

    -- Island state outputs → chiplets
    island_states_out : out island_state_array_t;

    -- PMIC power-good inputs
    pmic_pgood        : in  std_logic_vector(NUM_ISLANDS-1 downto 0);

    -- PMBus interface (I2C-like, open-drain)
    pmbus_scl         : inout std_logic;
    pmbus_sda         : inout std_logic;

    -- UART debug console
    uart_tx           : out std_logic;
    uart_rx           : in  std_logic
  );
end entity krafgpu_pmc;

architecture rtl of krafgpu_pmc is

  -- ===========================================================================
  -- PMC INTERNAL STATE MACHINE
  -- ===========================================================================
  type pmc_state_t is (
    PMC_INIT,           -- Startup: configure PMICs to safe voltages
    PMC_SAMPLE,         -- Read telemetry counters
    PMC_PREDICT,        -- EMA prediction of future demand
    PMC_DECIDE,         -- V/F decision per island
    PMC_ISSUE_PMBUS,    -- Send PMBus VOUT_COMMAND
    PMC_WAIT_PGOOD,     -- Wait for PMIC to confirm new voltage
    PMC_THERMAL_CHECK,  -- Guard: check TJ limits
    PMC_LOG,            -- UART log event (if change occurred)
    PMC_IDLE            -- Wait for next 1ms tick
  );

  signal pmc_state        : pmc_state_t := PMC_INIT;
  signal pmc_next_state   : pmc_state_t;

  -- ===========================================================================
  -- TIMING: 1ms control loop at 400 MHz = 400,000 cycles
  -- ===========================================================================
  constant LOOP_CYCLES    : integer := 400_000;
  signal   loop_timer     : integer range 0 to LOOP_CYCLES-1 := 0;
  signal   loop_tick      : std_logic := '0';

  -- ===========================================================================
  -- EMA FILTER STATE
  -- Exponential moving average: EMA_new = α × sample + (1-α) × EMA_old
  -- α = 0.125 (1/8) implemented as right-shift-3
  -- ===========================================================================
  type ema_array_t is array (0 to NUM_ISLANDS-1) of unsigned(13 downto 0);
  signal ema_util         : ema_array_t := (others => (others => '0'));

  -- ===========================================================================
  -- ISLAND STATE REGISTERS
  -- ===========================================================================
  signal island_states    : island_state_array_t :=
                            (others => ISLAND_FULL);

  -- ===========================================================================
  -- PMBUS CONTROLLER INTERFACE
  -- ===========================================================================
  signal pmbus_start      : std_logic := '0';
  signal pmbus_done       : std_logic;
  signal pmbus_island_id  : unsigned(3 downto 0);
  signal pmbus_voltage_mv : unsigned(11 downto 0);  -- Target voltage in mV
  signal pmbus_busy       : std_logic;

  -- ===========================================================================
  -- THERMAL GUARD
  -- ===========================================================================
  constant TEMP_LIMIT_C   : unsigned(7 downto 0) := to_unsigned(95, 8);
  constant TEMP_WARN_C    : unsigned(7 downto 0) := to_unsigned(85, 8);
  signal   thermal_event  : std_logic_vector(NUM_ISLANDS-1 downto 0);

  -- ===========================================================================
  -- PENDING STATE CHANGES
  -- ===========================================================================
  type pending_state_t is array (0 to NUM_ISLANDS-1) of island_state_t;
  signal pending_states   : pending_state_t := (others => ISLAND_FULL);
  signal island_idx       : integer range 0 to NUM_ISLANDS-1 := 0;
  signal change_pending   : std_logic_vector(NUM_ISLANDS-1 downto 0) := (others => '0');

  -- ===========================================================================
  -- UART LOG SIGNALS
  -- ===========================================================================
  signal uart_send        : std_logic := '0';
  signal uart_byte        : std_logic_vector(7 downto 0);
  signal uart_busy        : std_logic;

begin

  -- ===========================================================================
  -- 1ms LOOP TIMER
  -- ===========================================================================
  p_loop_timer : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        loop_timer <= 0;
        loop_tick  <= '0';
      elsif loop_timer = LOOP_CYCLES - 1 then
        loop_timer <= 0;
        loop_tick  <= '1';
      else
        loop_timer <= loop_timer + 1;
        loop_tick  <= '0';
      end if;
    end if;
  end process p_loop_timer;

  -- ===========================================================================
  -- PMC MAIN STATE MACHINE
  -- ===========================================================================
  p_pmc_fsm : process(clk)
    variable new_state : island_state_t;
    variable ema_new   : unsigned(13 downto 0);
    variable util_u14  : unsigned(13 downto 0);
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        pmc_state      <= PMC_INIT;
        island_idx     <= 0;
        pmbus_start    <= '0';
        change_pending <= (others => '0');
        for i in 0 to NUM_ISLANDS-1 loop
          island_states(i)  <= ISLAND_FULL;
          pending_states(i) <= ISLAND_FULL;
          ema_util(i)       <= (others => '0');
        end loop;

      else
        pmbus_start <= '0';
        uart_send   <= '0';

        case pmc_state is

          -- ---------------------------------------------------------------
          when PMC_INIT =>
            -- Set all islands to FULL on startup, wait one loop tick
            for i in 0 to NUM_ISLANDS-1 loop
              island_states(i) <= ISLAND_FULL;
            end loop;
            if loop_tick = '1' then
              pmc_state <= PMC_SAMPLE;
            end if;

          -- ---------------------------------------------------------------
          when PMC_SAMPLE =>
            -- Latch telemetry snapshot (single cycle)
            island_idx <= 0;
            pmc_state  <= PMC_PREDICT;

          -- ---------------------------------------------------------------
          when PMC_PREDICT =>
            -- EMA update for each island in sequence
            util_u14 := resize(telemetry_in(island_idx).utilisation, 14);

            -- EMA = EMA_old + (sample - EMA_old) >> 3
            if util_u14 > ema_util(island_idx) then
              ema_new := ema_util(island_idx) +
                         shift_right(util_u14 - ema_util(island_idx), 3);
            else
              ema_new := ema_util(island_idx) -
                         shift_right(ema_util(island_idx) - util_u14, 3);
            end if;
            ema_util(island_idx) <= ema_new;

            if island_idx < NUM_ISLANDS - 1 then
              island_idx <= island_idx + 1;
            else
              island_idx <= 0;
              pmc_state  <= PMC_DECIDE;
            end if;

          -- ---------------------------------------------------------------
          when PMC_DECIDE =>
            -- V/F decision for each island based on EMA utilisation
            for i in 0 to NUM_ISLANDS-1 loop

              -- Thermal override takes priority
              if telemetry_in(i).temperature >= TEMP_LIMIT_C then
                new_state := ISLAND_LOW;    -- Emergency throttle
                thermal_event(i) <= '1';
              elsif telemetry_in(i).temperature >= TEMP_WARN_C then
                new_state := ISLAND_HALF;   -- Warm throttle
                thermal_event(i) <= '0';
              -- Normal demand-based decision
              elsif ema_util(i)(6 downto 0) > to_unsigned(80, 7) then
                new_state := ISLAND_FULL;   -- >80% util → full perf
                thermal_event(i) <= '0';
              elsif ema_util(i)(6 downto 0) > to_unsigned(40, 7) then
                new_state := ISLAND_HALF;   -- 40-80% → balanced
                thermal_event(i) <= '0';
              elsif ema_util(i)(6 downto 0) > to_unsigned(5, 7) then
                new_state := ISLAND_LOW;    -- 5-40% → efficiency
                thermal_event(i) <= '0';
              else
                new_state := ISLAND_OFF;    -- <5% → gate off
                thermal_event(i) <= '0';
              end if;

              -- Record if change is needed
              if new_state /= island_states(i) then
                pending_states(i)  <= new_state;
                change_pending(i)  <= '1';
              else
                change_pending(i)  <= '0';
              end if;

            end loop;
            island_idx <= 0;
            pmc_state  <= PMC_ISSUE_PMBUS;

          -- ---------------------------------------------------------------
          when PMC_ISSUE_PMBUS =>
            -- Issue PMBus command for next island that needs change
            if change_pending(island_idx) = '1' and pmbus_busy = '0' then
              pmbus_island_id  <= to_unsigned(island_idx, 4);
              pmbus_voltage_mv <=
                to_unsigned(to_volt_mv(pending_states(island_idx)), 12);
              pmbus_start      <= '1';
              change_pending(island_idx) <= '0';
              pmc_state        <= PMC_WAIT_PGOOD;
            elsif island_idx < NUM_ISLANDS - 1 then
              island_idx <= island_idx + 1;
            else
              island_idx <= 0;
              pmc_state  <= PMC_THERMAL_CHECK;
            end if;

          -- ---------------------------------------------------------------
          when PMC_WAIT_PGOOD =>
            -- Wait for PMIC to assert PGOOD on this island
            if pmic_pgood(island_idx) = '1' and pmbus_busy = '0' then
              -- Voltage settled — update island state register
              island_states(island_idx) <= pending_states(island_idx);
              pmc_state <= PMC_ISSUE_PMBUS;
            end if;

          -- ---------------------------------------------------------------
          when PMC_THERMAL_CHECK =>
            -- Scan for any island exceeding TEMP_LIMIT_C
            for i in 0 to NUM_ISLANDS-1 loop
              if thermal_event(i) = '1' then
                island_states(i) <= ISLAND_LOW;
              end if;
            end loop;
            pmc_state <= PMC_LOG;

          -- ---------------------------------------------------------------
          when PMC_LOG =>
            -- Send a one-byte UART heartbeat: current active island count
            uart_byte <=
              std_logic_vector(to_unsigned(
                count_active_islands(island_states), 8));
            uart_send <= '1';
            pmc_state <= PMC_IDLE;

          -- ---------------------------------------------------------------
          when PMC_IDLE =>
            -- Wait for next 1ms tick
            if loop_tick = '1' then
              pmc_state <= PMC_SAMPLE;
            end if;

        end case;
      end if;
    end if;
  end process p_pmc_fsm;

  -- ===========================================================================
  -- OUTPUTS
  -- ===========================================================================
  island_states_out <= island_states;

  -- ===========================================================================
  -- PMBUS CONTROLLER (I2C master for PMIC voltage commands)
  -- ===========================================================================
  u_pmbus : entity work.krafgpu_pmbus_ctrl
    port map (
      clk          => clk,
      rst_n        => rst_n,
      start        => pmbus_start,
      island_id    => pmbus_island_id,
      voltage_mv   => pmbus_voltage_mv,
      busy         => pmbus_busy,
      done         => pmbus_done,
      scl          => pmbus_scl,
      sda          => pmbus_sda
    );

  -- ===========================================================================
  -- UART TRANSMITTER (debug console, 115200 baud)
  -- ===========================================================================
  u_uart : entity work.krafgpu_uart_tx
    generic map (
      CLK_FREQ_HZ  => 400_000_000,
      BAUD_RATE    => 115_200
    )
    port map (
      clk          => clk,
      rst_n        => rst_n,
      send         => uart_send,
      data         => uart_byte,
      busy         => uart_busy,
      tx           => uart_tx
    );

end architecture rtl;
