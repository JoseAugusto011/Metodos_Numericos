# **Métodos Numéricos para Resolução de Sistemas Lineares**

Este projeto apresenta implementações de métodos numéricos eficientes para resolver sistemas de equações lineares, utilizando **Python** e a biblioteca **NumPy** para otimização de cálculo.

## **Métodos Implementados**

✅ **Gauss-Seidel** (GaussSeidelSolver)  
✅ **TDMA** (Tridiagonal Matrix Algorithm) (TDMASolver)  
✅ **Decomposição LU** (LUSolver)  
✅ **Eliminação de Gauss** (GaussEliminationSolver)  

---

## **1. Gauss-Seidel (GaussSeidelSolver)**

### **Objetivo**
Resolve um sistema de equações lineares usando o **método iterativo de Gauss-Seidel**. Indicado para matrizes **diagonalmente dominantes**.

### **Entradas**
- `A`: Matriz de coeficientes *(n x n)*
- `B`: Vetor de constantes *(n x 1)*
- `X0` (opcional): Chute inicial *(n x 1)* *(padrão: vetor de uns)*
- `tol` (opcional): Tolerância do erro *(padrão: 1e-6)*
- `max_iterations` (opcional): Número máximo de iterações *(padrão: 1000)*

### **Saídas**
- `X`: Vetor solução *(n x 1)*
- `iterations`: Número de iterações realizadas

### **Exemplo de Uso**
```python
A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
B = np.array([15, 10, 10])
solver = GaussSeidelSolver(A, B)
X, iterations = solver.solve()
print(f"Solução: {X}")
print(f"Iterações: {iterations}")
```

---

## **2. TDMA (Tridiagonal Matrix Algorithm) (TDMASolver)**

### **Objetivo**
Resolve sistemas de equações onde a **matriz de coeficientes é tridiagonal**, garantindo alta eficiência computacional.

### **Entradas**
- `A`: Matriz tridiagonal de coeficientes *(n x n)*
- `B`: Vetor de constantes *(n x 1)*

### **Saídas**
- `X`: Vetor solução *(n x 1)*

### **Exemplo de Uso**
```python
A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
B = np.array([15, 10, 10])
solver = TDMASolver(A, B)
X = solver.solve()
print(f"Solução: {X}")
```

---

## **3. Decomposição LU (LUSolver)**

### **Objetivo**
Resolve um sistema de equações lineares usando a **decomposição LU**, que transforma a matriz de coeficientes em **L** (matriz triangular inferior) e **U** (matriz triangular superior), permitindo resolver o sistema com substituição direta e reversa.

### **Entradas**
- `A`: Matriz de coeficientes *(n x n)*
- `B`: Vetor de constantes *(n x 1)*

### **Saídas**
- `X`: Vetor solução *(n x 1)*
- `L`: Matriz triangular inferior *(n x n)*
- `U`: Matriz triangular superior *(n x n)*

### **Exemplo de Uso**
```python
A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
B = np.array([15, 10, 10])
solver = LUSolver(A, B)
X, L, U = solver.solve()
print(f"Solução: {X}")
print(f"Matriz L:\n{L}")
print(f"Matriz U:\n{U}")
```

---

## **4. Eliminação de Gauss (GaussEliminationSolver)**

### **Objetivo**
Resolve sistemas lineares utilizando o **método direto da eliminação de Gauss**, transformando a matriz de coeficientes em uma **matriz triangular superior** e resolvendo o sistema por substituição reversa.

### **Entradas**
- `A`: Matriz de coeficientes *(n x n)*
- `B`: Vetor de constantes *(n x 1)*

### **Saídas**
- `X`: Vetor solução *(n x 1)*

### **Exemplo de Uso**
```python
A = np.array([[1, 2], [4, 5]])
B = np.array([-1, 4])
solver = GaussEliminationSolver(A, B)
X = solver.solve()
print(f"Solução: {X}")
```

---

## **Requisitos**

Para rodar este projeto, você precisará de:

- **Python 3.x**
- **Biblioteca NumPy** *(instalável via pip:)*
  ```bash
  pip install numpy
  ```

---

## **Como Executar**

1️⃣ Clone este repositório ou copie o código das classes.  
2️⃣ Instale as dependências:  
   ```bash
   pip install numpy
   ```
3️⃣ Execute os exemplos de cada método.

---

## **Considerações Finais**

✔ **Eficiência**: Implementação otimizada com **operações vetorizadas do NumPy**.  
✔ **Robustez**: Verificações de consistência, como **matriz quadrada** e **compatibilidade de dimensões**.  
✔ **Documentação**: Código bem comentado para facilitar o uso e entendimento.  

🚀 **Agora é só testar e resolver seus sistemas lineares com eficiência!**

