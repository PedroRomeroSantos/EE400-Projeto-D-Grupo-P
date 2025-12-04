import numpy as np
from scipy.optimize import newton

#constantes
u = 3.986004418e5 #parâmetro gravitacional
v = 300000 # c = velocidade da luz (km/s)
#tempo marcado pelo receptor agora (ms)
# para que TOF = TOA - TOT seja positivo, TOA deve ser maior que TOT
#Fica mais fácil de entender as contas
TOA = 600000

#Elementos Orbitais para cada satélite
satelites = {
    'satelite_1': {'a': 15300, 'e': 0.41, 'w': 60, 'i': 30, 'o': 0, 'dt': 4708.5603},
    'satelite_2': {'a': 16100, 'e': 0.342, 'w': 10, 'i': 30, 'o': 40, 'dt': 5082.6453},
    'satelite_3': {'a': 17800, 'e': 0.235, 'w': 30, 'i': 0, 'o': 40, 'dt': 5908.5511},
    'satelite_4': {'a': 16400, 'e': 0.3725, 'w': 60, 'i': 20, 'o': 40, 'dt': 5225.3666}
}

#rotações
def rotacao_z(alpha):
    return np.array([[np.cos(alpha), -np.sin(alpha), 0],
                     [np.sin(alpha), np.cos(alpha), 0],
                     [0, 0, 1]])

def rotacao_x(alpha):
    return np.array([[1, 0, 0],
                     [0, np.cos(alpha), -np.sin(alpha)],
                     [0, np.sin(alpha), np.cos(alpha)]])

def posicao(sat):
    #extrai os elementos orbitais dos satélites
    a = sat['a']
    e = sat['e']
    w = np.radians(sat['w'])
    i = np.radians(sat['i'])
    o = np.radians(sat['o'])
    dt = sat['dt']

    #período (T) e anomalia média (M_e)
    T = 2 * np.pi * np.sqrt(a**3 / u)
    M_e = 2 * np.pi * dt / T

    #resolvendo Kepler
    def kepler(E, M_e, e):
        return E - e * np.sin(E) - M_e

    def dist_kepler(E, M_e, e):
        return 1 - e * np.cos(E)

    #resolve E com Newton-Raphson
    E = newton(func=kepler, fprime=dist_kepler, x0=np.pi, args=(M_e, e))

    #sistema perifocal
    xk = a * (np.cos(E) - e)
    yk = a * np.sin(E) * np.sqrt(1 - e**2)
    pos_perifocal = np.array([xk, yk, 0])

    #transformação para o sistema ECI, 
    # o @ é o operador de multiplicação matricial
    R = rotacao_z(o) @ rotacao_x(i) @ rotacao_z(w)
    return R @ pos_perifocal

# posições dos satélites
#precisamos saber onde eles estão ANTES de simular os tempos
lista_r = [posicao(sat) for sat in satelites.values()]

#criamos o TOT artificialmente pois não temos dados reais.
# Ex: Um ponto no equador com raio da terra ~6371km
posicao_real_drone = np.array([-6420., -6432., 6325.]) 

TOT = {}

for i, pos_sat in enumerate(lista_r):
    #distância exata entre satélite e drone real
    dist_real = np.linalg.norm(pos_sat - posicao_real_drone)
    
    #tempo de voo em segundos
    tof_s = dist_real / v 
    
    #gerar o TOT (ms)
    tot_ms = TOA - (tof_s * 1000)
    
    TOT[f'tempo_{i+1}'] = tot_ms
    #print(f"Sat {i+1}: Distancia={dist_real:.2f}km, TOT gerado={tot_ms:.4f}ms")

#descobrindo a posição real com os TOTs e as posições dos satélites.
# Calcular o TOF percebido (s) baseado nos dados recebidos
TOF = [(TOA - TOT[f'tempo_{i+1}']) / 1000 for i in range(len(satelites))]

def gradiente(lista_r, r, TOF):
    gradient = np.zeros(3)
    for i, pos_sat in enumerate(lista_r):
        p = v * TOF[i]
        distancia_vetor = r - pos_sat
        modulo = np.linalg.norm(distancia_vetor)
        R = 1 - (p / modulo)
        gradient += R * distancia_vetor
    return gradient

#chute inicial (qualquer na superfície da terra)
r_estimado = np.array([-6371., 0., 0.]) 

print(f"\nIniciando Otimização...")
print(f"Chute inicial: {r_estimado}")

#loop de otimização (Gradiente Descendente)
learning_rate = 0.6
for i in range(500): # 500 iterações
    G = gradiente(lista_r, r_estimado, TOF)
    r_estimado = r_estimado - learning_rate * G

print("\n--- Resultado ---")
print("Posição Real (Definida):   ", posicao_real_drone)
print("Posição Estimada (Calculada):", r_estimado)
erro = np.linalg.norm(posicao_real_drone - r_estimado)
print(f"Erro final: {erro:.4f} km")