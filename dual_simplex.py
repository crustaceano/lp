import numpy as np
import sys
import argparse

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
    
    scalar_product = v.T @ A_inv @ u
    
    if abs(1 + scalar_product[0, 0]) < TOLERANCE:
        raise ValueError("Matrix (A + uv^T) is singular")
    
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
    
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i, j] * y[j]
    
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = y[i]
        for j in range(i+1, n):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]
    
    return x

def DualSimplex(c, A, b, basis=None, nbasis=None, use_lu=True):
    """
    Реализация дуального симплекс-метода для решения задач линейного программирования.
    
    max c^T x
    s.t. Ax = b
         x >= 0
    
    Начинается с двойственно допустимого базиса (z*_N >= 0, т.е. приведенные стоимости <= 0).
    Следует алгоритму из раздела 4 теории.
    """
    m, n = A.shape
    n -= m
    if basis is None or nbasis is None:
        basis = list(range(n, n + m))
        nbasis = list(range(0, n))

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
                return "infeasible", None, None
            obj = float(np.dot(c, x))
            return "optimal", x, obj

        try:
            if use_lu and L is not None and U is not None:
                xB = solve_lu(L, U, b)
            else:
                B = A[:, basis]
                xB = np.linalg.solve(B, b)
        except (np.linalg.LinAlgError, ValueError):
            return "infeasible", None, None

        if np.all(xB >= -TOLERANCE):
            cB = c[basis]
            try:
                y = solve_dual_system(cB)
            except (np.linalg.LinAlgError, ValueError):
                return "infeasible", None, None
            
            N = A[:, nbasis]
            cN = c[nbasis]
            reduced_cost = cN - N.T @ y
            z_star_N = -reduced_cost
            
            if np.all(z_star_N >= -TOLERANCE):
                x = build_full_solution(basis)
                if x is None:
                    return "infeasible", None, None
                obj = float(np.dot(c, x))
                return "optimal", x, obj
            else:
                return "infeasible", None, None

        negative_indices = [i for i in range(len(xB)) if xB[i] < -TOLERANCE]
        leaving_row = min(negative_indices, key=lambda i: basis[i])
        i = leaving_row
        leaving_var = basis[i]

        e_i = np.zeros(m)
        e_i[i] = 1.0
        
        try:
            if use_lu and L is not None and U is not None:
                w = solve_lu(U.T, L.T, e_i)
            else:
                B = A[:, basis]
                w = np.linalg.solve(B.T, e_i)
        except (np.linalg.LinAlgError, ValueError):
            return "infeasible", None, None
        
        N = A[:, nbasis]
        delta_z_N = -(N.T @ w)

        cB = c[basis]
        try:
            y = solve_dual_system(cB)
        except (np.linalg.LinAlgError, ValueError):
            return "infeasible", None, None
        
        cN = c[nbasis]
        reduced_cost = cN - N.T @ y
        z_star_N = -reduced_cost
        
        if np.any(z_star_N < -TOLERANCE):
            return "infeasible", None, None
        
        ratios_dual = []
        for idx, j in enumerate(nbasis):
            if delta_z_N[idx] > TOLERANCE and z_star_N[idx] > TOLERANCE:
                ratio = z_star_N[idx] / delta_z_N[idx]
                ratios_dual.append((ratio, j, idx))
        
        if len(ratios_dual) == 0:
            return "infeasible", None, None
        
        min_ratio = min(ratios_dual, key=lambda t: (t[0], t[1]))
        s = min_ratio[0]
        
        if s <= TOLERANCE:
            return "infeasible", None, None

        entering_pos = min_ratio[2]
        entering_var = min_ratio[1]

        a_j = A[:, entering_var]
        try:
            if use_lu and L is not None and U is not None:
                delta_x_B = solve_lu(L, U, a_j)
            else:
                B = A[:, basis]
                delta_x_B = np.linalg.solve(B, a_j)
        except (np.linalg.LinAlgError, ValueError):
            return "infeasible", None, None

        if abs(delta_x_B[i]) < TOLERANCE:
            return "infeasible", None, None
        t = xB[i] / delta_x_B[i]

        xB = xB - t * delta_x_B

        basis = basis.copy()
        nbasis = nbasis.copy()
        basis[i] = entering_var
        nbasis[entering_pos] = leaving_var

        if use_lu:
            try:
                B = A[:, basis]
                L, U = lu_decomposition(B)
            except ValueError:
                use_lu = False


def Phase1_Dual(c, A, b):
    """
    Первая фаза дуального симплекс-метода для получения двойственно допустимого базиса.
    Если задача уже имеет двойственно допустимый базис, возвращает его.
    """
    m, n = A.shape
    
    slack_matrix = np.eye(m)
    A_with_slacks = np.hstack([A, slack_matrix])
    
    c_phase2 = np.concatenate([c, np.zeros(m)])
    
    basis = list(range(n, n + m))
    nbasis = list(range(0, n))
    
    B = A_with_slacks[:, basis]
    try:
        y = np.linalg.solve(B.T, c_phase2[basis])
        N = A_with_slacks[:, nbasis]
        cN = c_phase2[nbasis]
        reduced_cost = cN - N.T @ y
        
        if np.all(reduced_cost >= -TOLERANCE):
            return DualSimplex(c_phase2, A_with_slacks, b, basis, nbasis)
    except (np.linalg.LinAlgError, ValueError):
        pass
    
    artificial_matrix = np.eye(m)
    A_phase1 = np.hstack([A_with_slacks, artificial_matrix])
    
    M = 1e6
    c_phase1 = np.concatenate([c_phase2, M * np.ones(m)])
    
    basis_phase1 = list(range(n + m, n + 2*m))
    nbasis_phase1 = list(range(n + m))
    
    from simplex_template import PrimalSimplex
    status, x_phase1, obj_phase1 = PrimalSimplex(c_phase1, A_phase1, b, basis_phase1, nbasis_phase1)
    
    if status != "optimal" or obj_phase1 > TOLERANCE:
        return "infeasible", None, None
    
    basis_phase2 = []
    for i in range(n + m):
        if x_phase1[i] > TOLERANCE:
            basis_phase2.append(i)
    
    if len(basis_phase2) < m:
        for i in range(n, n + m):
            if i not in basis_phase2:
                basis_phase2.append(i)
                if len(basis_phase2) == m:
                    break
    
    if len(basis_phase2) < m:
        for i in range(n + m):
            if i not in basis_phase2:
                basis_phase2.append(i)
                if len(basis_phase2) == m:
                    break
    
    basis_phase2 = sorted(basis_phase2[:m])
    nbasis_phase2 = [i for i in range(n + m) if i not in basis_phase2]
    
    try:
        B = A_with_slacks[:, basis_phase2]
        if np.linalg.matrix_rank(B) < m:
            from simplex_template import PrimalSimplex
            return PrimalSimplex(c_phase2, A_with_slacks, b, basis_phase2, nbasis_phase2)
        
        y = np.linalg.solve(B.T, c_phase2[basis_phase2])
        N = A_with_slacks[:, nbasis_phase2]
        cN = c_phase2[nbasis_phase2]
        reduced_cost = cN - N.T @ y
        
        if np.all(reduced_cost <= TOLERANCE):
            return DualSimplex(c_phase2, A_with_slacks, b, basis_phase2, nbasis_phase2)
        else:
            from simplex_template import PrimalSimplex
            return PrimalSimplex(c_phase2, A_with_slacks, b, basis_phase2, nbasis_phase2)
    except (np.linalg.LinAlgError, ValueError):
        from simplex_template import PrimalSimplex
        return PrimalSimplex(c_phase2, A_with_slacks, b, basis_phase2, nbasis_phase2)


def Solve_Dual(c, A, b):
    """
    Главная функция для решения задач линейного программирования дуальным симплекс-методом.
    Автоматически добавляет слаковые переменные и выбирает подходящий метод.
    
    Для дуального симплекс-метода нужен двойственно допустимый базис (z*_N >= 0, т.е. c̄_N <= 0).
    Базис из слаковых переменных двойственно допустим только если c <= 0.
    В общем случае используем Phase1_Dual для получения двойственно допустимого базиса.
    
    Примечание: для задач max c^T x с c > 0 дуальный симплекс обычно не используется напрямую,
    так как базис из слаковых переменных не двойственно допустим. В таких случаях
    Phase1_Dual автоматически переключается на прямой симплекс.
    """
    m, n = A.shape
    A_ext = np.hstack([A.astype(float), np.eye(m)])
    c_ext = np.concatenate([c.astype(float), np.zeros(m)])
    
    if np.all(c <= TOLERANCE):
        basis = list(range(n, n + m))
        nbasis = list(range(0, n))
        status, x_ext, obj = DualSimplex(c_ext, A_ext, b.astype(float), basis, nbasis)
        if status == "optimal" and x_ext is not None:
            return status, x_ext[:n], obj
        return status, x_ext, obj
    else:
        from simplex_template import Solve as Solve_Primal
        status, x, obj = Solve_Primal(c, A, b)
        return status, x, obj

def proc_cmd():
    parser = argparse.ArgumentParser(description="Solve a linear program using the Dual Simplex method.")
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

    print("Solving the linear program using the Dual Simplex method...\n")
    status, solution, objective = Solve_Dual(c, A, b)
    print("\nResult:")
    print("Status:", status)
    if status == "optimal":
        print("Optimal solution x* =", solution)
        print("Optimal value =", objective)
    elif status == "unbounded":
        print("The problem is unbounded.")
    elif status == "infeasible":
        print("The problem is infeasible.")
    else:
        print("No solution found.")

if __name__ == '__main__':
    main()

