def parse_ampl_routes(ampl_text, cities):
    """Parse salesman routes from AMPL ``display x`` output.

    AMPL may print ``x`` either as individual assignments ``x[s,i,j] = 1`` or as
    matrices.  This parser supports both formats and handles multiple salesmen.
    """

    edges = []

    # 1) ``x[s,i,j] = 1`` assignments
    pattern = re.compile(r"x\[(\d+),(\d+),(\d+)\]\s*=\s*1")
    for s, i, j in pattern.findall(ampl_text):
        edges.append((int(s), int(i), int(j)))

    # 2) Matrix form ``x [s,*,*]`` possibly written as ``[s,*,*]``
    matrix_pat = re.compile(r"(?:x\s*)?\[(\d+),\*,\*\]")
    lines = ampl_text.splitlines()
    idx = 0
    while idx < len(lines):
        m = matrix_pat.match(lines[idx].strip())
        if not m:
            idx += 1
            continue
        s = int(m.group(1))
        idx += 1
        if idx >= len(lines):
            break
        header = lines[idx]
        cols = [int(n) for n in re.findall(r"-?\d+", header)]
        idx += 1
        while idx < len(lines):
            row = lines[idx].strip()
            if not row or row.startswith("[") or row.startswith(";"):
                break
            tokens = re.findall(r"-?\d+", row)
            if len(tokens) == len(cols) + 1:
                i = int(tokens[0])
                for col, val in zip(cols, tokens[1:]):
                    if val == "1":
                        edges.append((s, i, col))
            idx += 1
        # do not consume delimiter line; outer loop will reconsider it
    if not edges:
        return []

    salesmen = sorted({s for s, _, _ in edges})
    by_s = {s: {} for s in salesmen}
    for s, i, j in edges:
        by_s[s][i] = j

    city_map = {c.idx: c for c in cities}
    routes = []

    for s in salesmen:
        if s not in by_s:
            continue
        route = [city_map[0]]  # Empezar en el depósito
        current = 0
        visited = set()

        while True:
            if current in visited:
                break  # Evitar ciclos infinitos
            visited.add(current)

            nxt = by_s[s].get(current)
            if nxt is None or nxt == current:
                break

            if nxt == 0:
                route.append(city_map[0])
                break

            route.append(city_map[nxt])
            current = nxt

            if current == 0 or len(visited) > len(city_map):
                break

        if route[-1].idx != 0:
            route.append(city_map[0])

        routes.append(route)

    return routes
