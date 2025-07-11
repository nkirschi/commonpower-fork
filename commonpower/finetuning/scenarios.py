from abc import ABCMeta, abstractmethod
from pathlib import Path

from utils import Approach, CEnum, Penalty, Stage

from commonpower.control.controllers import *
from commonpower.control.runners import DeploymentRunner
from commonpower.control.safety_layer.penalties import *
from commonpower.control.safety_layer.safety_layers import *
from commonpower.core import ModelHistory, System
from commonpower.data_forecasting.base import DataProvider, Forecaster
from commonpower.data_forecasting.data_sources import *
from commonpower.data_forecasting.forecasters import *
from commonpower.modeling.param_initialization import *
from commonpower.models.buses import *
from commonpower.models.components import *
from commonpower.models.powerflow import *


class BaseScenario(metaclass=ABCMeta):
    def __init__(self, mode: Stage, forecaster: Forecaster, data_path: Path, date_format: str):
        super().__init__()
        self.mode = mode  # 'training' or 'deployment'
        self.data_path = data_path.resolve()
        self.date_format = date_format
        self.forecaster = forecaster
        self.forecast_frequency = forecaster.frequency
        self.forecast_horizon = forecaster.horizon
        self.sys = self.create_system()

    def get_system(self) -> System:
        return self.sys

    def create_system(self) -> System:
        return self._create_system()

    @abstractmethod
    def _create_system(self) -> System:
        pass


def create_scenario(
    stage: Stage,
    scenario_constructor: BaseScenario,
    approach: Approach,
    penalty: Penalty,
    forecast_length: int,
    forecaster: Forecaster,
):
    forecast_horizon = timedelta(hours=forecast_length)
    current_path = Path().absolute()
    data_path = current_path / 'data'
    date_format = "%Y-%m-%d %H:%M:00"

    # penalty configs
    constant_penalty = 1.0
    penalty_factor = 1.0

    # safeguard, penalty
    if approach in [Approach.WithProjectionSafeguard, Approach.WithReplacementSafeguard]:
        if penalty is Penalty.NoPenalty:
            safety_penalty = NoPenalty()
            penalty_factor = 0.0
            constant_penalty = 0.0
        elif penalty is Penalty.ConstantPenalty:
            safety_penalty = ConstantPenalty(penalty_constant=constant_penalty)
            penalty_factor = 0.0
        elif penalty is Penalty.DDPenalty:
            safety_penalty = DistanceDependingPenalty(penalty_factor=penalty_factor)
            constant_penalty = 0.0
        else:
            safety_penalty = DistanceDependingPenalty(penalty_factor=penalty_factor, penalty_constant=constant_penalty)
        if approach is Approach.WithProjectionSafeguard:
            safeguard = ActionProjectionSafetyLayer(penalty=safety_penalty)
        else:
            safeguard = ActionReplacementWithOptSafetyLayer(penalty=safety_penalty)
    elif approach is Approach.OptimalController:
        safeguard = None
    else:
        raise NotImplementedError

    # create system
    train_scenario = scenario_constructor(
        mode=stage,
        data_path=data_path,
        date_format=date_format,
        forecaster=forecaster,
        use_heat_pump=False,
    )

    sys = train_scenario.get_system()

    if approach is Approach.OptimalController:
        controller = OptimalController(name="agent1")
    else:
        controller = RLController(
            name="agent1",
            safety_layer=safeguard,
            obs_handler=ObservationHandler(num_forecasts=forecast_length),
        )
    controller.add_entity(sys.nodes[0])

    # Create deployment runner
    oc_history = ModelHistory([sys])
    eval_seed = 5
    oc_deployer = DeploymentRunner(sys=sys, horizon=forecast_horizon, history=oc_history, seed=eval_seed)

    return sys, oc_deployer


class BuildingManagementSystemScenario(BaseScenario):
    def __init__(
        self,
        mode: Stage,
        forecaster: Forecaster,
        data_path: Path,
        date_format: str,
        use_heat_pump: bool = False,
        price_buying: float = 0.37,
        price_selling: float = 0.08,  # numbers for 2024: https://www.finanztip.de/photovoltaik/einspeiseverguetung/
    ):
        self.use_heat_pump = use_heat_pump
        self.price_buying = price_buying
        self.price_selling = price_selling
        super().__init__(mode=mode, forecaster=forecaster, data_path=data_path, date_format=date_format)

    def _create_system(self):
        self.define_data_sources()
        self.load_p_dp = DataProvider(self.p_load_ds, self.forecaster)  # [kW]
        self.load_q_dp = DataProvider(self.q_load_ds, self.forecaster)  # [kVA]
        self.price_dp = DataProvider(self.buying_price_ds, self.forecaster)  # [€]
        self.selling_price_dp = DataProvider(self.selling_price_ds, self.forecaster)  # [€]
        self.pv_dp = DataProvider(self.pv_ds, self.forecaster)  # [kW]
        self.hp_dp = DataProvider(self.heat_pump_ds, self.forecaster)
        # Let's first create an instance of the RTPricedBus with lower and upper bounds for its variables
        n1 = RTPricedBus("MultiFamilyHouse", {'p': (-50, 50), 'q': (-50, 50), 'v': (0.95, 1.05), 'd': (-15, 15)})
        # Then, we add the previously defined data providers for the buying and selling price of electricity.
        n1.add_data_provider(self.price_dp).add_data_provider(self.selling_price_dp)

        # external grid
        m1 = ExternalGrid("ExternalGrid")

        # photovoltaic with generation data
        r1 = RenewableGen("PV1").add_data_provider(self.pv_dp)

        # static load with data source
        d1 = Load("Load1").add_data_provider(self.load_p_dp).add_data_provider(self.load_q_dp)

        # battery storage
        capacity = 5  # kWh
        # if self.mode is Stage.Train:
        #     ess_initializer = RangeInitializer(0.2 * capacity, 0.8 * capacity)
        #     hp_initializer = RangeInitializer(18, 24)
        # else:
        #     # during deployment, we want a fixed initial soc
        #     ess_initializer = ConstantInitializer(0.5 * capacity)
        #     hp_initializer = ConstantInitializer(21)
        # Since it has proven beneficial for training to use constant initializers, we change this:
        ess_initializer = ConstantInitializer(0.5 * capacity)
        hp_initializer = ConstantInitializer(21)

        e1 = ESSLinear(
            "ESS1",
            {
                'p': (-1.5, 1.5),  # active power limits
                'q': (0, 0),  # reactive power limits
                'soc': (0.1 * capacity, 0.9 * capacity),  # soc limits
                "soc_init": ess_initializer,
            },
        )

        h1 = HeatPumpWithoutStorageButCOP(
            "HeatPump",
            {
                'p': [0, 5],  # kW
                'T_indoor_setpoint': 21,  # Celsius
                'T_indoor': [16, 26],  # Celsius
                'T_indoor_init': hp_initializer,  # Celsius
                'T_ret_FH': [10, 100],  # Celsius
                'T_ret_FH_init': ConstantInitializer(25.0),  # Celsius
                'H_FH': 1.1,  # kW/K
                'H_out': 0.26,  # kW/K
                'tau_building': 240,  # h
                'Cw_FH': 1.1625,  # kWh/K
                'c': 1.0,  # weighting factor (comfort factor) for cost function; multiplied with temperature deviation
            },
        ).add_data_provider(self.hp_dp)

        # add components to the household
        n1.add_node(d1).add_node(r1).add_node(e1)
        if self.use_heat_pump:
            n1.add_node(h1)

        # create the system and add top-level busses
        return System(power_flow_model=PowerBalanceModel()).add_node(n1).add_node(m1)

    def define_data_sources(self):
        # Data source (ds) for active power(p) of a household
        self.p_load_ds = CSVDataSource(
            self.data_path / 'ICLR_load.csv', datetime_format=self.date_format, resample=self.forecast_frequency
        )

        # We neglect reactive power (q) during this tutorial
        self.q_load_ds = ConstantDataSource(
            {"q": 0.0}, date_range=self.p_load_ds.get_date_range(), frequency=self.forecast_frequency
        )

        self.buying_price_ds = ConstantDataSource(
            {"psib": self.price_buying},
            date_range=self.p_load_ds.get_date_range(),
            frequency=self.forecast_frequency,
        )

        # Data source for selling prices of electricity
        self.selling_price_ds = ConstantDataSource(
            {"psis": self.price_selling},
            date_range=self.buying_price_ds.get_date_range(),
            frequency=self.forecast_frequency,
        )

        # Data source for PV generation
        self.pv_ds = CSVDataSource(
            self.data_path / 'ICLR_pv.csv', datetime_format=self.date_format, resample=self.forecast_frequency
        ).apply_to_column("p", lambda x: -x)

        # Data sources for heat pump: outdoor temperature and coefficient of performance
        # taken from When2Heat dataset: https://data.open-power-system-data.org/when2heat/
        self.heat_pump_ds = CSVDataSource(
            self.data_path / 'DE_Temperature_and_COP2016.csv',
            datetime_format="%d.%m.%Y %H:%M",
            rename_dict={"time": "t", "outside_temp": "T_outside", "COP": "COP"},
            auto_drop=True,
            delimiter=";",
            resample=self.forecast_frequency,
        )


class BuildingManagementSystemToUPricesScenario(BuildingManagementSystemScenario):
    def __init__(
        self,
        mode: Stage,
        forecaster: Forecaster,
        data_path: Path,
        date_format: str,
        use_heat_pump: bool = False,
        price_selling: float = 0.08,
    ):
        super().__init__(
            mode=mode,
            forecaster=forecaster,
            data_path=data_path,
            date_format=date_format,
            use_heat_pump=use_heat_pump,
            price_selling=price_selling,
        )

    def define_data_sources(self):
        # Data source (ds) for active power(p) of a household
        self.p_load_ds = CSVDataSource(
            self.data_path / 'ICLR_load.csv', datetime_format=self.date_format, resample=self.forecast_frequency
        )

        # We neglect reactive power (q) during this tutorial
        self.q_load_ds = ConstantDataSource(
            {"q": 0.0}, date_range=self.p_load_ds.get_date_range(), frequency=self.forecast_frequency
        )

        # Time of Use prices with three levels, time slots accoring to
        # https://publica-rest.fraunhofer.de/server/api/core/bitstreams/be2409c6-eb92-4e32-b4be-81983bed8ec1/content
        self.buying_price_ds = CSVDataSource(
            self.data_path / 'ToU_prices.csv', datetime_format=self.date_format, resample=self.forecast_frequency
        )

        # Data source for selling prices of electricity
        self.selling_price_ds = ConstantDataSource(
            {"psis": self.price_selling},
            date_range=self.buying_price_ds.get_date_range(),
            frequency=self.forecast_frequency,
        )

        # Data source for PV generation
        self.pv_ds = CSVDataSource(
            self.data_path / 'ICLR_pv.csv', datetime_format=self.date_format, resample=self.forecast_frequency
        ).apply_to_column("p", lambda x: -x)

        # Data sources for heat pump: outdoor temperature and coefficient of performance
        # taken from When2Heat dataset: https://data.open-power-system-data.org/when2heat/
        self.heat_pump_ds = CSVDataSource(
            self.data_path / 'DE_Temperature_and_COP2016.csv',
            datetime_format="%d.%m.%Y %H:%M",
            rename_dict={"time": "t", "outside_temp": "T_outside", "COP": "COP"},
            auto_drop=True,
            delimiter=";",
            resample=self.forecast_frequency,
        )


class BuildingManagementSystemWithEVScenario(BuildingManagementSystemScenario):
    def __init__(
        self,
        mode: Stage,
        forecaster: Forecaster,
        data_path: Path,
        date_format: str,
        use_heat_pump: bool = False,
        price_buying: float = 0.37,
        price_selling: float = 0.08,  # numbers for 2024: https://www.finanztip.de/photovoltaik/einspeiseverguetung/
    ):
        super().__init__(
            mode=mode,
            forecaster=forecaster,
            data_path=data_path,
            date_format=date_format,
            use_heat_pump=use_heat_pump,
            price_buying=price_buying,
            price_selling=price_selling,
        )

    def define_data_sources(self):
        # Data source (ds) for active power(p) of a household WITH ELECTRIC VEHICLE
        self.p_load_ds = CSVDataSource(
            self.data_path / 'ICLR_load_with_ev.csv', datetime_format=self.date_format, resample=self.forecast_frequency
        )
        # We neglect reactive power (q) during this tutorial
        self.q_load_ds = ConstantDataSource(
            {"q": 0.0}, date_range=self.p_load_ds.get_date_range(), frequency=self.forecast_frequency
        )

        self.buying_price_ds = ConstantDataSource(
            {"psib": self.price_buying},
            date_range=self.p_load_ds.get_date_range(),
            frequency=self.forecast_frequency,
        )

        # Data source for selling prices of electricity
        self.selling_price_ds = ConstantDataSource(
            {"psis": self.price_selling},
            date_range=self.buying_price_ds.get_date_range(),
            frequency=self.forecast_frequency,
        )

        # Data source for PV generation
        self.pv_ds = CSVDataSource(
            self.data_path / 'ICLR_pv.csv', datetime_format=self.date_format, resample=self.forecast_frequency
        ).apply_to_column("p", lambda x: -x)

        # Data sources for heat pump: outdoor temperature and coefficient of performance
        # taken from When2Heat dataset: https://data.open-power-system-data.org/when2heat/
        self.heat_pump_ds = CSVDataSource(
            self.data_path / 'DE_Temperature_and_COP2016.csv',
            datetime_format="%d.%m.%Y %H:%M",
            rename_dict={"time": "t", "outside_temp": "T_outside", "COP": "COP"},
            auto_drop=True,
            delimiter=";",
            resample=self.forecast_frequency,
        )


class Scenario(CEnum):
    ConstantPricesScenario = BuildingManagementSystemScenario
    ToUPricesScenario = BuildingManagementSystemToUPricesScenario
    AddedEVScenario = BuildingManagementSystemWithEVScenario
