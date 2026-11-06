"""
Gráficas del estudio de Optuna para la presentación.

- optuna_pong.py las genera automáticamente al terminar.
- También se puede ejecutar a mano (p. ej. a mitad del estudio):
      python optuna_plots.py

Genera HTML interactivos en optuna_plots/ (abrir con: explorer.exe optuna_plots).
Necesita: pip install plotly
"""
import os
import optuna
from optuna.visualization import (
    plot_param_importances,
    plot_optimization_history,
    plot_intermediate_values,
    plot_parallel_coordinate,
    plot_slice,
    plot_timeline,
)

OUT = "optuna_plots"


def make_plots(study):
    os.makedirs(OUT, exist_ok=True)
    states = optuna.trial.TrialState
    done = [t for t in study.trials if t.state == states.COMPLETE]
    pruned = [t for t in study.trials if t.state == states.PRUNED]
    print(f"Pruebas: {len(study.trials)} | completas: {len(done)} | podadas: {len(pruned)}")

    plots = {
        "1_historia": lambda: plot_optimization_history(study),
        "2_curvas_y_poda": lambda: plot_intermediate_values(study),
        "3_coordenadas_paralelas": lambda: plot_parallel_coordinate(study),
        "4_slice": lambda: plot_slice(study),
        # La importancia necesita al menos 2 pruebas completas
        "5_importancia": lambda: plot_param_importances(study),
        # Duración de cada prueba y tiempo ahorrado por el pruning
        "6_timeline": lambda: plot_timeline(study),
    }
    for name, make in plots.items():
        try:
            make().write_html(f"{OUT}/{name}.html")
            print(f"  ok: {OUT}/{name}.html")
        except Exception as e:
            print(f"  no se pudo generar {name}: {e}")

    if done:
        print(f"\nMejor prueba: #{study.best_trial.number} | score {study.best_value:.2f}")
        print(f"Parámetros: {study.best_params}")

    print("\nResumen por prueba:")
    for t in study.trials:
        last = max(t.intermediate_values) if t.intermediate_values else 0
        val = f"{t.value:.2f}" if t.value is not None else "-"
        print(f"  #{t.number:02d} {t.state.name:9s} hasta {last:>7} pasos | "
              f"score {val:>6} | {t.params}")


if __name__ == "__main__":
    study = optuna.load_study(study_name="ppo_pong",
                              storage="sqlite:///optuna_pong.db")
    make_plots(study)
