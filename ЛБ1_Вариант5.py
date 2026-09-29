"""
Лабораторная работа № 1. Вариант 5.
Итерационные методы вычисления sqrt(a), cbrt(b), ln(c).
Критерий остановки для корней: по разности |x_new - x| < eps.
Дополнительное требование Д5: график N(eps) для eps = 1e-1 .. 1e-10.
"""
import math
import matplotlib.pyplot as plt

A, B, C = 11.0, 20.0, 1.5
EPS = 1e-7
N_MAX = 1000


def sqrt_newton(a, eps=EPS, n_max=N_MAX, track=False):
    """Квадратный корень методом Ньютона (формула Герона). x0 = a."""
    if a < 0:
        raise ValueError("Подкоренное выражение должно быть неотрицательным")
    if a == 0:
        return (0.0, 0, []) if track else (0.0, 0)
    x = a
    history = []
    for n in range(1, n_max + 1):
        x_new = 0.5 * (x + a / x)
        diff = abs(x_new - x)
        if track:
            history.append((n, x, x_new, diff))
        if diff < eps:
            return (x_new, n, history) if track else (x_new, n)
        x = x_new
    raise RuntimeError("Точность не достигнута за N_MAX итераций")


def cbrt_newton(b, eps=EPS, n_max=N_MAX, track=False):
    """Кубический корень методом Ньютона. x0 = |b|, знак восстанавливается."""
    if b == 0:
        return (0.0, 0, []) if track else (0.0, 0)
    sign = -1 if b < 0 else 1
    b_abs = abs(b)
    x = b_abs
    history = []
    for n in range(1, n_max + 1):
        x_new = (2 * x + b_abs / (x * x)) / 3
        diff = abs(x_new - x)
        if track:
            history.append((n, sign * x, sign * x_new, diff))
        if diff < eps:
            return (sign * x_new, n, history) if track else (sign * x_new, n)
        x = x_new
    raise RuntimeError("Точность не достигнута за N_MAX итераций")


def ln_series(c, eps=EPS, n_max=100_000, track=False):
    """ln c через ряд ln((1+y)/(1-y)), y=(c-1)/(c+1)."""
    if c <= 0:
        raise ValueError("Аргумент логарифма должен быть положительным")
    y = (c - 1) / (c + 1)
    y2 = y * y
    power = y
    total = 0.0
    history = []
    for k in range(n_max):
        term = 2 * power / (2 * k + 1)
        if track:
            history.append((k, total, term, abs(term)))
        if abs(term) < eps:
            return (total, k, history) if track else (total, k)
        total += term
        power *= y2
    raise RuntimeError("Точность не достигнута за n_max итераций")


def print_main_table():
    print(f"{'Функция':<8}{'Арг.':>8}{'Результат':>18}"
          f"{'Эталон':>18}{'Погрешность':>15}{'Итераций':>10}")
    rows = [
        ("sqrt",  sqrt_newton, A, math.sqrt(A)),
        ("cbrt",  cbrt_newton, B, B ** (1 / 3)),
        ("ln",    ln_series,   C, math.log(C)),
    ]
    for name, f, arg, ref in rows:
        val, n = f(arg, EPS)
        print(f"{name:<8}{arg:>8}{val:>18.12f}{ref:>18.12f}"
              f"{abs(val - ref):>15.2e}{n:>10}")


def print_iterations():
    """Подробная таблица итераций (для отчёта)."""
    print("\n--- Таблица итераций (eps = 1e-7) ---")
    for name, f, arg in [("sqrt(11)", sqrt_newton, A),
                         ("cbrt(20)", cbrt_newton, B),
                         ("ln(1.5)",  ln_series,  C)]:
        _, n, hist = f(arg, EPS, track=True)
        print(f"\n{name}: всего {n} итераций")
        print(f"{'n':>4}{'x_n':>20}{'x_(n+1)':>20}{'|dx|':>16}")
        for row in hist:
            print(f"{row[0]:>4}{row[1]:>20.12f}{row[2]:>20.12f}{row[3]:>16.3e}")


def epsilon_study(eps_list):
    """Возвращает словарь {метод: [число итераций для каждого eps]}."""
    res = {"sqrt": [], "cbrt": [], "ln": []}
    for eps in eps_list:
        res["sqrt"].append(sqrt_newton(A, eps)[1])
        res["cbrt"].append(cbrt_newton(B, eps)[1])
        res["ln"].append(ln_series(C, eps)[1])
    return res


def plot_N_eps(eps_list, res):
    """Д5: график N(eps) в полулогарифмическом масштабе."""
    xs = [math.log10(e) for e in eps_list]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, res["sqrt"], "o-", label="sqrt(a) (Ньютон)")
    plt.plot(xs, res["cbrt"], "s-", label="cbrt(b) (Ньютон)")
    plt.plot(xs, res["ln"],   "^-", label="ln c (ряд)")
    plt.xlabel("log10 eps")
    plt.ylabel("Число итераций N")
    plt.title("Зависимость N(eps) для трёх методов (вариант 5)")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("N_eps.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    print(f"Вариант 5: a={A}, b={B}, c={C}, eps={EPS}\n")
    print_main_table()
    print_iterations()

    eps_list = [10.0 ** (-k) for k in range(1, 11)]
    res = epsilon_study(eps_list)
    print("\n--- Д5: N(eps) ---")
    print(f"{'eps':>10}{'sqrt':>8}{'cbrt':>8}{'ln':>8}")
    for e, ns, nc, nl in zip(eps_list, res["sqrt"], res["cbrt"], res["ln"]):
        print(f"{e:>10.0e}{ns:>8}{nc:>8}{nl:>8}")
    plot_N_eps(eps_list, res)