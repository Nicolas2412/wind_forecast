import os
import pandas as pd
import torch
from main import main  # Remplace par le nom de ton fichier

# 📋 Grille de recherche
MODEL_TYPES = ["lstm"]
SEQ_LENGTHS = [6]
DROP_PROD_OPTIONS = [False]

def run_automated_training(site:int):
    results_file = f"data/models/grid_search_summary_site_{site}.csv"

    for m_type in MODEL_TYPES:
        for seq in SEQ_LENGTHS:
            for dp in DROP_PROD_OPTIONS:
                suffix = "no_prod" if dp else "with_prod"
                model_name = f"{m_type}_seq{seq}_{suffix}"

                print(f"\n🚀 Lancement : {model_name}")

                # Le 'main' doit retourner les métriques pour le log
                # Si ton main ne retourne rien, il faudra le modifier légèrement
                try:
                    metrics = main(
                        model_type=m_type,
                        seq_len=seq,
                        drop_prod=dp,
                        no_cv=False,  # ✅ On garde la CV
                        savepath=f"data/models/{m_type}/site{site}/{model_name}.pkl",
                        test_size=0.2,
                        one_site_only=True,
                        idx_site=site
                    )
                    torch.cuda.empty_cache()
                    # Log des résultats dans un CSV
                    if metrics:
                        log_results(results_file, model_name,
                                    m_type, seq, dp, metrics)

                except Exception as e:
                    print(f"❌ Erreur sur {model_name} : {e}")


def log_results(file_path, name, m_type, seq, dp, metrics):
    data = {
        "name": [name],
        "model": [m_type],
        "seq_len": [seq],
        "no_prod": [dp],
        "mae": [metrics['eval_mae']],
        "rmse": [metrics['eval_rmse']],
        "nrmse_pct": [metrics['eval_nrmse'] * 100],
        "portfolio_mae_total": [metrics.get('portfolio_mae_total', None)],
        "portfolio_rmse_total": [metrics.get('portfolio_rmse_total', None)],
        "portfolio_nrmse_total": [metrics.get('portfolio_nrmse_total', None)],
        "portfolio_mae_per_site": [metrics.get('portfolio_mae_per_site', None)],
        "portfolio_rmse_per_site": [metrics.get('portfolio_rmse_per_site', None)],
    }
    df = pd.DataFrame(data)
    if not os.path.isfile(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)


if __name__ == "__main__":
    for i in range(3,10):
        print("STARTING GRID SEARCH FOR SITE", i)
        run_automated_training(site=i)
