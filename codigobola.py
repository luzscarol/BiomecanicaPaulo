import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Definir o caminho do vídeo
video_path = "C:\\Users\\carol\\Downloads\\IBmBiomec\\Tarefa2\\c1.avi"

# Carregar o vídeo
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Erro ao abrir o vídeo. Verifique o caminho do arquivo.")
    exit()

# Lista para armazenar as coordenadas 2D da bola em cada frame
coordinates_2d = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converter para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar suavização
    gray = cv2.GaussianBlur(gray, (11, 11), 0)  # Reduzir o tamanho do kernel para uma suavização mais leve

    # Aplicar limite para reduzir ruídos
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    # Detectar círculos usando a Transformada de Hough com parâmetros ajustados
    circles = cv2.HoughCircles(
        thresh, cv2.HOUGH_GRADIENT, dp=1.5, minDist=30,
        param1=70, param2=20, minRadius=8, maxRadius=35
    )
    
    # Se houver círculos detectados
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        x, y, r = circles[0]  # Pegando o primeiro círculo encontrado
        coordinates_2d.append((x, y))

        # Desenhar o círculo detectado para visualização
        cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
    
    # Mostrar o frame com a bola detectada (opcional)
    cv2.imshow("Detecção da Bola", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Pressione 'q' para sair manualmente
        break

cap.release()
cv2.destroyAllWindows()

# Verificar se temos dados de coordenadas 2D suficientes
if len(coordinates_2d) == 0:
    print("Nenhum círculo foi detectado nos frames do vídeo.")
    exit()

# Converter coordinates_2d para um array Numpy
coordinates_2d = np.array(coordinates_2d)

# Criar um DataFrame do pandas a partir das coordenadas
df = pd.DataFrame(coordinates_2d, columns=['x', 'y'])

# Definir o caminho para salvar o CSV
csv_file_path = "C:\\Users\\carol\\Downloads\\coordenadas_bola.csv"
df.to_csv(csv_file_path, index=False)

print(f"Coordenadas salvas em {csv_file_path}")

# Plotar a trajetória da bola em 2D
plt.figure()
plt.plot(coordinates_2d[:, 0], coordinates_2d[:, 1], 'o-', color='blue')
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.title("Trajetória da Bola em 2D")
plt.gca().invert_yaxis()  # Inverter o eixo Y para corresponder à orientação da imagem
plt.show()

# falta ajustar o reconhecimento de círculos