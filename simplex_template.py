import numpy as np
import sys
import argparse

# Константы
TOLERANCE = 1e-5

def sherman_morrison_update(A_inv, u, v):
    """
    Обновление обратной матрицы по формуле Шермана-Моррисона.
    Вычисляет (A + uv^T)^(-1) через A^(-1) без полного пересчета.
    
    Args:
        A_inv: обратная матрица A^(-1)
        u, v: векторы для рангового обновления A + uv^T
    
    Returns:
        обновленная обратная матрица (A + uv^T)^(-1)
    """
    u = np.array(u).reshape(-1, 1)
    v = np.array(v).reshape(-1, 1)
    
    # Вычисляем скалярное произведение v^T A^(-1) u
    scalar_product = v.T @ A_inv @ u
    
    # Проверка на сингулярность
    if abs(1 + scalar_product[0, 0]) < TOLERANCE:
        raise ValueError("Matrix (A + uv^T) is singular")
    
    # Применяем формулу Шермана-Моррисона
    Ainv_u = A_inv @ u
    vT_Ainv = v.T @ A_inv
    
    updated_inv = A_inv - (Ainv_u @ vT_Ainv) / (1 + scalar_product[0, 0])
    
    return updated_inv

def lu_decomposition(A):
    """
    LU-разложение квадратной матрицы A = L * U
    
    Args:
        A: квадратная матрица n x n
    
    Returns:
        L, U: нижняя и верхняя треугольные матрицы
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")
    
    L = np.eye(n)
    U = A.copy()
    
    for k in range(n-1):
        if abs(U[k, k]) < TOLERANCE:
            raise ValueError("Matrix is singular or near-singular")
        
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            for j in range(k, n):
                U[i, j] -= L[i, k] * U[k, j]
    
    return L, U

def solve_lu(L, U, b):
    """
    Решение системы Ax = b через LU-разложение
    
    Args:
        L, U: LU-разложение матрицы A
        b: правая часть
    
    Returns:
        x: решение системы
    """
    b = np.array(b, dtype=float)
    n = len(b)
    
    # Прямая подстановка: Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i, j] * y[j]
    
    # Обратная подстановка: Ux = y
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = y[i]
        for j in range(i+1, n):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]
    
    return x

def PrimalSimplex(c, A, b, basis=None, nbasis=None, use_lu=True):
    """
    Реализация прямого симплекс-метода для решения задач линейного программирования.
    
    max c^T x
    s.t. Ax = b
         x >= 0
    """
    m, n = A.shape
    n -= m
    if basis is None or nbasis is None:
        basis = list(range(n, n + m))
        nbasis = list(range(0, n))

    # Инициализация LU-разложения
    L, U = None, None
    if use_lu:
        try:
            B = A[:, basis]
            L, U = lu_decomposition(B)
        except ValueError:
            use_lu = False

    def build_full_solution(current_basis):
        B = A[:, current_basis]
        try:
            if use_lu and L is not None and U is not None:
                xB = solve_lu(L, U, b)
            else:
                xB = np.linalg.solve(B, b)
        except (np.linalg.LinAlgError, ValueError):
            return None
        x = np.zeros(A.shape[1])
        x[current_basis] = xB
        return x

    def solve_dual_system(cB):
        try:
            if use_lu and L is not None and U is not None:
                z = solve_lu(U.T, L.T, cB)
                y = solve_lu(L.T, U.T, z)
                return y
            else:
                B = A[:, basis]
                return np.linalg.solve(B.T, cB)
        except (np.linalg.LinAlgError, ValueError):
            raise

    max_iters = 10000
    iters = 0
    while True:
        iters += 1
        if iters > max_iters:
            x = build_full_solution(basis)
            if x is None:
                return "unbounded", None, None
            obj = float(np.dot(c, x))
            return "optimal", x, obj

        # Вычисление потенциалов
        cB = c[basis]
        try:
            y = solve_dual_system(cB)
        except (np.linalg.LinAlgError, ValueError):
            return "unbounded", None, None

        # Вычисление приведенных стоимостей
        N = A[:, nbasis]
        cN = c[nbasis]
        reduced_cost = cN - N.T @ y

        # Выбор входящей переменной по правилу Бленда
        candidates = [(j, idx) for idx, j in enumerate(nbasis) if reduced_cost[idx] > TOLERANCE]
        if len(candidates) == 0:
            x = build_full_solution(basis)
            if x is None:
                return "unbounded", None, None
            obj = float(np.dot(c, x))
            return "optimal", x, obj

        entering_var = min(j for j, _ in candidates)
        entering_pos = nbasis.index(entering_var)

        # Вычисление направления
        a_e = A[:, entering_var]
        try:
            if use_lu and L is not None and U is not None:
                d = solve_lu(L, U, a_e)
            else:
                B = A[:, basis]
                d = np.linalg.solve(B, a_e)
        except (np.linalg.LinAlgError, ValueError):
            return "unbounded", None, None

        # Проверка на неограниченность
        positive_mask = d > TOLERANCE
        if not np.any(positive_mask):
            return "unbounded", None, None

        # Правило минимального отношения
        try:
            if use_lu and L is not None and U is not None:
                xB = solve_lu(L, U, b)
            else:
                B = A[:, basis]
                xB = np.linalg.solve(B, b)
        except (np.linalg.LinAlgError, ValueError):
            return "unbounded", None, None
        ratios = []
        for i_row, di in enumerate(d):
            if di > TOLERANCE:
                ratios.append((xB[i_row] / di, basis[i_row], i_row))
        if len(ratios) == 0:
            return "unbounded", None, None
        min_theta = min(ratios, key=lambda t: (t[0], t[1]))
        leaving_row = min_theta[2]
        leaving_var = basis[leaving_row]

        # Обновление базиса
        basis = basis.copy()
        nbasis = nbasis.copy()
        basis[leaving_row] = entering_var
        nbasis[entering_pos] = leaving_var

        # Обновление LU-разложения
        if use_lu:
            try:
                B = A[:, basis]
                L, U = lu_decomposition(B)
            except ValueError:
                use_lu = False


def Phase1(c, A, b):
    """
    Первая фаза симплекс-метода для поиска начального допустимого базиса.
    Добавляет искусственные переменные и минимизирует их сумму.
    """
    m, n = A.shape
    
    # Добавление слаковых переменных
    slack_matrix = np.eye(m)
    A_with_slacks = np.hstack([A, slack_matrix])
    
    # Добавление искусственных переменных
    artificial_matrix = np.eye(m)
    A_phase1 = np.hstack([A_with_slacks, artificial_matrix])
    
    # Целевая функция для минимизации суммы искусственных переменных
    c_phase1 = np.concatenate([np.zeros(n), np.zeros(m), -np.ones(m)])
    
    # Начальный базис из искусственных переменных
    basis = list(range(n + m, n + 2*m))
    nbasis = list(range(n + m))
    
    # Запуск первой фазы
    status, x_phase1, obj_phase1 = PrimalSimplex(c_phase1, A_phase1, b, basis, nbasis)
    
    # Проверка на несовместность
    if status != "optimal" or obj_phase1 > TOLERANCE:
        return "infeasible", None, None
    
    # Переход ко второй фазе
    A_phase2 = A_with_slacks
    c_phase2 = np.concatenate([c, np.zeros(m)])
    
    # Базис для второй фазы (без искусственных переменных)
    basis_phase2 = [i for i in basis if i < n + m]
    nbasis_phase2 = [i for i in range(n + m) if i not in basis_phase2]
    
    return PrimalSimplex(c_phase2, A_phase2, b, basis_phase2, nbasis_phase2)


def Solve(c, A, b):
    """
    Главная функция для решения задач линейного программирования.
    Автоматически добавляет слаковые переменные и выбирает подходящий метод.
    """
    m, n = A.shape
    A_ext = np.hstack([A.astype(float), np.eye(m)])
    c_ext = np.concatenate([c.astype(float), np.zeros(m)])
    
    if np.all(b >= 0):
        status, x_ext, obj = PrimalSimplex(c_ext, A_ext, b.astype(float))
        if status == "optimal" and x_ext is not None:
            return status, x_ext[:n], obj
        return status, x_ext, obj
    else:
        status, x_ext, obj = Phase1(c_ext, A_ext, b.astype(float))
        if status == "optimal" and x_ext is not None:
            return status, x_ext[:n], obj
        return status, x_ext, obj

def proc_cmd():
    parser = argparse.ArgumentParser(description="Solve a linear program using the Primal Simplex method.")
    parser.add_argument("filename", type=str, help="Input file containing the LP problem.")
    return parser.parse_args()

def test_sherman_morrison():
    print("=== Тест формулы Шермана-Моррисона ===")
    
    np.random.seed(42)
    A = np.random.rand(3, 3) + np.eye(3)
    A_inv = np.linalg.inv(A)
    
    u = np.array([1, 0, 0])
    v = np.array([0, 1, 0])
    
    A_new_inv = sherman_morrison_update(A_inv, u, v)
    
    A_new = A + np.outer(u, v)
    A_new_inv_direct = np.linalg.inv(A_new)
    
    error = np.linalg.norm(A_new_inv - A_new_inv_direct)
    print(f"Ошибка формулы Шермана-Моррисона: {error:.2e}")
    print(f"Формула работает корректно: {error < 1e-10}")
    print()

def test_lu_decomposition():
    print("=== Тест LU-разложения ===")
    
    np.random.seed(42)
    A = np.random.rand(4, 4) + np.eye(4)
    
    L, U = lu_decomposition(A)
    
    A_reconstructed = L @ U
    error = np.linalg.norm(A - A_reconstructed)
    print(f"Ошибка LU-разложения: {error:.2e}")
    print(f"LU-разложение работает корректно: {error < 1e-10}")
    
    b = np.array([1, 2, 3, 4])
    x_lu = solve_lu(L, U, b)
    x_direct = np.linalg.solve(A, b)
    
    error_solve = np.linalg.norm(x_lu - x_direct)
    print(f"Ошибка решения через LU: {error_solve:.2e}")
    print(f"Решение через LU работает корректно: {error_solve < 1e-10}")
    print()

def main():
    args = proc_cmd()
    
    test_sherman_morrison()
    test_lu_decomposition()
    
    with open(args.filename, 'r', encoding='utf-8') as f:
        n, m = map(int, f.readline().split())
        c = np.array(list(map(float, f.readline().split())))
        A = []
        b = []
        for _ in range(m):
            *row, bi = map(float, f.readline().split())
            A.append(row)
            b.append(bi)
        A = np.array(A)
        b = np.array(b)
    print("n =", n)
    print("m =", m)
    print("c =", c)
    print("A =", A)
    print("b =", b)

    print("Solving the linear program using the Primal Simplex method...\n")
    status, solution, objective = Solve(c, A, b)
    print("\nResult:")
    print("Status:", status)
    if status == "optimal":
        print("Optimal solution x* =", solution)
        print("Optimal value =", objective)
    elif status == "unbounded":
        print("The problem is unbounded.")
    else:
        print("No solution found.")

if __name__ == '__main__':
    main()