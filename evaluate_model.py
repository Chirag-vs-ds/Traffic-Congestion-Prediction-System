import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def evaluate(model, X_test, y_test):
    # ALWAYS create outputs folder safely
    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    file_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.title("Confusion Matrix")
    plt.savefig(file_path)
    plt.close()
