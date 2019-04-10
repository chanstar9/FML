from tqdm import tqdm

if __name__ == '__main__':
    ensemble_names = [
        'portfolio_blend-sharpe_ratio-True-3-0.0'
    ]

    outcomes_dict = {}

    for ensemble_name in tqdm(ensemble_names):
        outcomes = get_outcomes(ensemble_name)
        outcomes_dict[ensemble_name] = outcomes
