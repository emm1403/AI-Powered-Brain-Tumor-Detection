import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ruta al modelo entrenado y carpeta de test
model_path = 'modelo_mri_finetune.h5'
test_dir = 'C:/Users/DELL/Desktop/Proyecto LSC/IMAGES MRI/archive/Testing'

# Clases (deben coincidir con las carpetas en test_dir)
class_names = ['Astrocytoma', 'Glioma', 'Meningioma', 'Neurocytoma', 'Pituitary', 'Schwannoma', 'No Tumor']

# Cargar modelo
model = load_model(model_path)

# Generador de datos para test
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Predicción
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Reporte de clasificación
report = classification_report(y_true, y_pred, target_names=class_names)
print("Classification Report:\n", report)

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Gráfico de métricas por clase
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_true, y_pred, average=None)
recall = recall_score(y_true, y_pred, average=None)
f1 = f1_score(y_true, y_pred, average=None)

x = np.arange(len(class_names))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - width, precision, width=width, label='Precision', color='skyblue')
plt.bar(x, recall, width=width, label='Recall', color='lightgreen')
plt.bar(x + width, f1, width=width, label='F1-Score', color='salmon')

plt.xticks(x, class_names, rotation=45)
plt.ylim(0, 1.05)
plt.ylabel('Score')
plt.title('Per-Class Metrics')
plt.legend()
plt.tight_layout()
plt.savefig('per_class_metrics.png')
plt.show()
