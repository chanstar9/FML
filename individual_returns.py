import matplotlib.pyplot as plt
import pandas as pd
from ksif import *

from ensemble import get_predictions
from ensemble_method import select_predictions, concat_predictions, WEIGHT


def get_outcomes(model_name, quantile, long_only, start_number, end_number):
    predictions = get_predictions(model_name, start_number, end_number)
    if long_only:
        selected_predictions = select_predictions(predictions, quantile, True)
    else:
        long_predictions = select_predictions(predictions, quantile, True)
        short_predictions = select_predictions(predictions, quantile, False)
        selected_predictions = concat_predictions(long_predictions, short_predictions)

    selected_portfolios = [Portfolio(selected_prediction) for selected_prediction in selected_predictions]
    outcomes = [selected_portfolio.outcome(weighted=WEIGHT) for selected_portfolio in selected_portfolios]
    cumulative_returns = [selected_portfolio.get_returns(WEIGHT, cumulative=True) for selected_portfolio in
                          selected_portfolios]
    df_returns = pd.concat(cumulative_returns, axis=1, names=DATE)
    df_returns.columns = range(start_number, end_number + 1)
    title = "{}-{}_quantile-{}".format("Long_only" if long_only else "Long_short", quantile, model_name.split('-')[1])
    df_returns.plot(title=title)
    plt.savefig("individual_returns/{}.png".format(title))
    plt.show()

    return outcomes, title


if __name__ == '__main__':
    model_names = [
        'DNN8_3-value_size_momentum_quality-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-momentum-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-momentum_quality-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-quality-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-size-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-size_momentum-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-size_momentum_quality-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-size_quality-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-value-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-value_momentum-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-value_momentum_quality-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-value_quality-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-value_size-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-value_size_momentum-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-value_size_quality-tahn-zeros-lecun_normal-none-0.5',
    ]
    quantiles = [
        3, 5, 10
    ]
    for model_name in model_names:
        for quantile in quantiles:
            outcomes, title = get_outcomes(model_name, quantile, True, 0, 9)
            df_outcomes = pd.DataFrame({key: [outcome[key] for outcome in outcomes] for key in outcomes[0]})
            df_outcomes.to_csv("individual_returns/{}.csv".format(title))
