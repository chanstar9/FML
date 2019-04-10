import matplotlib.pyplot as plt
import pandas as pd
from ksif import *

from ensemble import get_predictions
from ensemble_method import *


def get_outcomes(method, model_name, quantile, long_only, start_number, end_number, per_ensemble):
    selected_portfolios = []
    for ensemble_num in range(start_number, end_number + 1, per_ensemble):
        predictions = get_predictions(model_name, ensemble_num, ensemble_num + per_ensemble - 1)
        get_ensemble_predictions = GET_ENSEMBLE_PREDICTIONS[method]
        ensemble_predictions = get_ensemble_predictions(predictions, quantile, long_only, NONE, 0)
        # Append actual returns
        final_ensemble_prediction = pd.merge(ensemble_predictions[-1], actual_long_returns, on=[DATE, CODE])
        selected_portfolios.append(Portfolio(final_ensemble_prediction))

    outcomes = [selected_portfolio.outcome(weighted=WEIGHT) for selected_portfolio in selected_portfolios]
    cumulative_returns = [selected_portfolio.get_returns(WEIGHT, cumulative=True) for selected_portfolio in
                          selected_portfolios]
    df_returns = pd.concat(cumulative_returns, axis=1, names=DATE)
    df_returns.columns = range(0, (end_number - start_number + 1) // per_ensemble)
    title = "{}-{}-{}_quantile-{}".format(
        "Long_only" if long_only else "Long_short", method, quantile, model_name.split('-')[1]
    )
    df_returns.plot(title=title)
    plt.savefig("simple_ensemble_returns/{}.png".format(title))
    plt.show()

    return outcomes, title


if __name__ == '__main__':
    model_names = [
        'DNN8_3-value_size_momentum_quality-tahn-zeros-lecun_normal-none-0.5',
    ]
    quantiles = [
        3, 5, 10
    ]
    for method in [PORTFOLIO_BLEND, SIGNAL_BLEND]:
        for model_name in model_names:
            for quantile in quantiles:
                outcomes, title = get_outcomes(method, model_name, quantile, True, 0, 99, 10)
                df_outcomes = pd.DataFrame({key: [outcome[key] for outcome in outcomes] for key in outcomes[0]})
                df_outcomes.to_csv("simple_ensemble_returns/{}.csv".format(title))
