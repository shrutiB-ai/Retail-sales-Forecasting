import matplotlib.pyplot as plt

def plot_predictions(y_test,y_pred_lin,y_pred_xgb,save_path=None):
    plt.figure(figsize=(10,5))
    plt.plot(y_test.values,label ='Actual',linestyle='--',marker ='o')
    plt.plot(y_pred_lin,label='Linear Pred',linestyle='-',marker='x')
    plt.plot(y_pred_xgb,label='XGBoost Pred',linestyle='-',marker='s')
    plt.title('Actual vs Prediction')
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
