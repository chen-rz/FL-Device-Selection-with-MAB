import math
import random
from logging import DEBUG, WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import flwr.common
import pandas
import torchvision
from flwr.common import (
    EvaluateIns, EvaluateRes, FitIns, FitRes, GetParametersIns,
    ndarrays_to_parameters, parameters_to_ndarrays,
    MetricsAggregationFn, NDArrays, Parameters, Scalar,
)
from flwr.common.logger import log
from flwr.server import SimpleClientManager
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy.strategy import Strategy

import client as clt
from constants import *
from dataset_utils import cifar10Transformation


class MAB_ClientManager(SimpleClientManager):
    def sample(self, num_clients: int, server_round=0, time_constr=0):
        # For model initialization
        if num_clients == 1:
            return [self.clients[str(random.randint(0, pool_size - 1))]]

        # For evaluation, use the same devices as in the fit round
        elif num_clients == -1:
            with open(
                    "./output/fit_server/round_{}.txt".format(server_round),
                    mode='r'
            ) as inputFile:
                cids_in_fit = eval(inputFile.readline())["clients_selected"]
            return [self.clients[str(cid)] for cid in cids_in_fit]

        # Sample clients which meet the criterion
        param_dicts = []
        available_cids = []

        C_record = []
        updateTimeList = []

        for n in range(pool_size):
            # Get each client's parameters
            param_dicts.append(
                self.clients[str(n)].get_properties(
                    flwr.common.GetPropertiesIns(config={}), 68400
                ).properties.copy()
            )

            param_dicts[n]["isSelected"] = False
            updateTimeList.append(param_dicts[n]["updateTime"])

        C_min = min(updateTimeList)
        C_max = max(updateTimeList)
        for n in range(pool_size):
            param_dicts[n]["C"] = (param_dicts[n]["updateTime"] - C_min) / (C_max - C_min)
            C_record.append(param_dicts[n]["C"])
        
        with open("./output/C_records/round_{}.txt".format(server_round), mode='w') as outputFile:
            outputFile.write(str(C_record))
        
        log(DEBUG, "Wrote C_record: " + str(C_record))

        # 1st iteration: data size only
        if server_round == 1:
            for i in range(pool_size):
                param_dicts[i]["D"] = param_dicts[i]["dataSize"]

            available_cids = sorted(
                range(pool_size), key=lambda i: param_dicts[i]["dataSize"], reverse=True
            )[:num_to_choose]

        # Common cases
        else:
            with open("./output/involvement_history.txt", mode='r') as inputFile:
                involvement_history = eval(inputFile.readline())

            log(DEBUG, "Involvement history: " + str(involvement_history))

            with open("./output/fit_server/round_{}.txt".format(server_round - 1)) \
                    as inputFile:
                cids_in_prev_round = eval(inputFile.readline())["clients_selected"]

            log(DEBUG, "Cids in previous round: " + str(cids_in_prev_round))

            loss_of_prev_round = []
            sum_of_loss_of_prev = 0.0
            for n in range(pool_size):
                with open("./output/train_loss/client_{}.txt".format(n)) as inputFile:
                    loss_of_prev_round.append(eval(inputFile.readlines()[-1]))
                if n in cids_in_prev_round:
                    assert loss_of_prev_round[-1] > 0
                    sum_of_loss_of_prev += loss_of_prev_round[-1]
                else:
                    assert loss_of_prev_round[-1] == -1
            for i in range(pool_size):
                if i not in cids_in_prev_round:
                    assert loss_of_prev_round[i] == -1
                    # For those did not involve in previous rounds, loss should be the average
                    loss_of_prev_round[i] = sum_of_loss_of_prev / len(cids_in_prev_round)

            log(DEBUG, "Loss in previous round: " + str(loss_of_prev_round))

            for i in range(pool_size):
                param_dicts[i]["D"] = param_dicts[i]["dataSize"] \
                     * loss_of_prev_round[i] / (involvement_history[i] + 1)
                
            sum_of_prev_C = [0 for _ in range(pool_size)]
            for t in range(1, server_round):
                with open("./output/fit_server/round_{}.txt".format(t)) as inputFile:
                    cids_in_t_round = eval(inputFile.readline())["clients_selected"]
                with open("./output/C_records/round_{}.txt".format(t)) as inputFile:
                    C_in_t_round = eval(inputFile.readline())
                for _ in range(pool_size):
                    if _ in cids_in_t_round:
                        sum_of_prev_C[_] += C_in_t_round[_]

            log(DEBUG, "Sum of C in previous round: " + str(sum_of_prev_C))
            
            UCB_mu = []
            for i in range(pool_size):
                if involvement_history[i] == 0:
                    UCB_mu.append(0)
                else:
                    UCB_mu.append(sum_of_prev_C[i] / involvement_history[i])

            log(DEBUG, "UCB_mu: " + str(UCB_mu))

            UCB_u = []
            for i in range(pool_size):
                if involvement_history[i] == 0:
                    UCB_u.append(0)
                else:
                    UCB_u.append(
                        UCB_mu[i] + math.sqrt(
                            (num_to_choose + 1) * math.log(server_round) / involvement_history[i]
                        )
                    )
                
            log(DEBUG, "UCB_U: " + str(UCB_u))

            UCB_omega = [
                -UCB_u[i] - beta * math.pow(math.e, (-param_dicts[i]["D"])) \
                for i in range(pool_size)
            ]

            log(DEBUG, "UCB_omega: " + str(UCB_omega))

            available_cids = sorted(
                range(pool_size), key=lambda i: UCB_omega[i], reverse=True
            )[:num_to_choose]

        # Record client parameters
        fit_round_time = 0
        for _ in available_cids:
            param_dicts[_]["isSelected"] = True
            if param_dicts[_]["updateTime"] > fit_round_time:
                fit_round_time = param_dicts[_]["updateTime"]

        # Record reward
        reward = 0
        for k in available_cids:
            reward += (-param_dicts[k]["C"] - beta * math.pow(math.e, (-param_dicts[k]["D"])))
        reward *= (1 / num_to_choose)
        with open("./output/reward.txt", mode='a') as outputFile:
            outputFile.write(str(reward) + "\n")

        log(DEBUG, "Round " + str(server_round) + " selected cids " + str(available_cids))
        log(DEBUG, "Round " + str(server_round) + " reward: " + str(reward))

        return [self.clients[str(cid)] for cid in available_cids], \
            {
                "clients_selected": available_cids,
                "time_elapsed": fit_round_time,
                "time_constraint": time_constr
            }, \
            param_dicts


class Random_ClientManager(SimpleClientManager):
    def sample(self, num_clients: int, server_round=0, time_constr=0):
        # For model initialization
        if num_clients == 1:
            return [self.clients[str(random.randint(0, pool_size - 1))]]

        # For evaluation
        elif num_clients == -1:
            with open(
                    "./output/fit_server/round_{}.txt".format(server_round),
                    mode='r'
            ) as inputFile:
                cids_in_fit = eval(inputFile.readline())["clients_selected"]
            return [self.clients[str(cid)] for cid in cids_in_fit]

        # Sample clients in a random way
        param_dicts = []

        cid_num = num_to_choose

        cids_tbd = list(range(pool_size))
        for _ in range(pool_size - cid_num):
            pop_idx = random.randint(0, len(cids_tbd) - 1)
            cids_tbd.pop(pop_idx)

        available_cids = cids_tbd.copy()
        assert len(available_cids) == cid_num

        for n in range(pool_size):
            # Get each client's parameters
            param_dicts.append(
                self.clients[str(n)].get_properties(
                    flwr.common.GetPropertiesIns(config={}), 68400
                ).properties.copy()
            )

            param_dicts[n]["isSelected"] = False

        fit_round_time = 0
        for _ in available_cids:
            param_dicts[_]["isSelected"] = True
            if param_dicts[_]["updateTime"] > fit_round_time:
                fit_round_time = param_dicts[_]["updateTime"]

        return [self.clients[str(cid)] for cid in available_cids], \
            {
                "clients_selected": available_cids,
                "time_elapsed": fit_round_time,
                "time_constraint": time_constr
            }, \
            param_dicts


class BSFL(Strategy):
    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(
            self,
            *,
            evaluate_fn: Optional[
                Callable[
                    [int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:

        super().__init__()

        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def __repr__(self) -> str:
        rep = f"TCS (accept_failures={self.accept_failures})"
        return rep

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
            self, server_round: int, parameters: Parameters,
            client_manager: Union[MAB_ClientManager, Random_ClientManager]
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit_clients config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Check records of previous round
        if server_round > 1:
            with open(
                    "./output/fit_server/round_{}.txt".format(server_round - 1),
                    mode='r'
            ) as inputFile:
                clients_of_prev_round = eval(inputFile.readline())["clients_selected"]

            for _ in range(pool_size):
                # If the client was not selected in the previous round,
                # help it complete the records
                if _ not in clients_of_prev_round:
                    with open(
                            "./output/train_loss/client_{}.txt".format(_),
                            mode='a'
                    ) as outputFile:
                        outputFile.write("-1" + "\n")
                    with open(
                            "./output/val_accu/client_{}.txt".format(_),
                            mode='a'
                    ) as outputFile:
                        outputFile.write("-1" + "\n")
                    with open(
                            "./output/val_loss/client_{}.txt".format(_),
                            mode='a'
                    ) as outputFile:
                        outputFile.write("-1" + "\n")

            # Record historic involvements
            with open("./output/involvement_history.txt", mode='r') as inputFile:
                fileLine = inputFile.readline()
                if not fileLine:
                    involvement_history = [0 for _ in range(pool_size)]
                else:
                    involvement_history = eval(fileLine)
            for _ in range(pool_size):
                if _ in clients_of_prev_round:
                    involvement_history[_] += 1
            with open("./output/involvement_history.txt", mode='w') as outputFile:
                outputFile.write(str(involvement_history))

        # Time constraint
        # if server_round == 1:
        #     C_T_i = timeConstrGlobal / num_rounds
        # else:
        #     with open(
        #             "./output/fit_server/round_{}.txt".format(server_round - 1)
        #     ) as inputFile:
        #         fit_round_dict = eval(inputFile.readline())
        #         C_T_i = timeConstrGlobal / num_rounds + \
        #                 fit_round_dict["time_constraint"] - fit_round_dict["time_elapsed"]

        # Sample clients
        clients, fit_round_dict, param_dicts = client_manager.sample(
            num_clients=0, server_round=server_round, time_constr=timeConstrGlobal
        )

        # Record information of clients
        pandas.DataFrame.from_records(param_dicts).to_excel(
            "./output/fit_clients/fit_round_{}.xlsx".format(server_round)
        )

        # Record information of server
        with open(
                "./output/fit_server/round_{}.txt".format(server_round),
                mode='w'
        ) as outputFile:
            outputFile.write(str(fit_round_dict))

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
            self, server_round: int, parameters: Parameters,
            client_manager: Union[MAB_ClientManager, Random_ClientManager]
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients: use same clients as in fit
        clients = client_manager.sample(
            num_clients=-1, server_round=server_round, time_constr=timeConstrGlobal
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit_clients results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated
