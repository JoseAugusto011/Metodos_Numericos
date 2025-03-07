Métodos Numéricos para Resolução de Sistemas Lineares

Este projeto contém implementações de métodos numéricos para resolver sistemas de equações lineares. Cada método é implementado como uma classe em Python, utilizando a biblioteca numpy para eficiência e desempenho.
Métodos Implementados

    Gauss-Seidel (GaussSeidelSolver)

    TDMA (Tridiagonal Matrix Algorithm) (TDMASolver)

    Decomposição LU (LUSolver)

    Eliminação de Gauss (GaussEliminationSolver)

1. Gauss-Seidel (GaussSeidelSolver)
Finalidade:

Resolve um sistema de equações lineares usando o método iterativo de Gauss-Seidel. É adequado para sistemas onde a matriz de coeficientes é diagonalmente dominante.
Entrada:

    A: Matriz de coeficientes (n x n).

    B: Vetor de constantes (n x 1).

    X0 (opcional): Chute inicial para a solução (n x 1). Padrão: vetor de uns.

    tol (opcional): Tolerância para o erro na solução. Padrão: 1e-6.

    max_iterations (opcional): Número máximo de iterações. Padrão: 1000.

Saída:

    X: Vetor solução (n x 1).

    iterations: Número de iterações realizadas.

Exemplo de Uso:
python
Copy

A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
B = np.array([15, 10, 10])
solver = GaussSeidelSolver(A, B)
X, iterations = solver.solve()
print(f"Solução: {X}")
print(f"Iterações: {iterations}")

2. TDMA (Tridiagonal Matrix Algorithm) (TDMASolver)
Finalidade:

Resolve sistemas de equações lineares onde a matriz de coeficientes é tridiagonal. Este método é altamente eficiente para esse tipo de sistema.
Entrada:

    A: Matriz tridiagonal de coeficientes (n x n).

    B: Vetor de constantes (n x 1).

Saída:

    X: Vetor solução (n x 1).

Exemplo de Uso:
python
Copy

A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
B = np.array([15, 10, 10])
solver = TDMASolver(A, B)
X = solver.solve()
print(f"Solução: {X}")

3. Decomposição LU (LUSolver)
Finalidade:

Resolve um sistema de equações lineares usando a decomposição LU. Este método decompõe a matriz de coeficientes em uma matriz triangular inferior (L) e uma matriz triangular superior (U), e então resolve o sistema usando substituição direta e reversa.
Entrada:

    A: Matriz de coeficientes (n x n).

    B: Vetor de constantes (n x 1).

Saída:

    X: Vetor solução (n x 1).

    L: Matriz triangular inferior (n x n).

    U: Matriz triangular superior (n x n).

Exemplo de Uso:
python
Copy

A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
B = np.array([15, 10, 10])
solver = LUSolver(A, B)
X, L, U = solver.solve()
print(f"Solução: {X}")
print(f"Matriz L:\n{L}")
print(f"Matriz U:\n{U}")

4. Eliminação de Gauss (GaussEliminationSolver)
Finalidade:

Resolve um sistema de equações lineares usando o método direto de eliminação de Gauss. Transforma a matriz de coeficientes em uma matriz triangular superior e resolve o sistema usando substituição reversa.
Entrada:

    A: Matriz de coeficientes (n x n).

    B: Vetor de constantes (n x 1).

Saída:

    X: Vetor solução (n x 1).

Exemplo de Uso:
python
Copy

A = np.array([[1, 2], [4, 5]])
B = np.array([-1, 4])
solver = GaussEliminationSolver(A, B)
X = solver.solve()
print(f"Solução: {X}")

Requisitos

    Python 3.x

    Biblioteca numpy (instalável via pip install numpy).

Como Executar

    Clone o repositório ou copie o código das classes.

    Instale as dependências:
    bash
    Copy

    pip install numpy

    Execute os exemplos fornecidos em cada seção.

Considerações Finais

    Eficiência: Todos os métodos foram implementados com foco em desempenho, utilizando operações vetorizadas do numpy.

    Robustez: Verificações de consistência (como matriz quadrada e compatibilidade de dimensões) são realizadas em cada método.

    Documentação: Cada classe e método está bem documentado para facilitar o uso e a compreensão.

