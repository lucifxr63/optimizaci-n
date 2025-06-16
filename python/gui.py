def parse_ampl_routes(ampl_text, cities):
    """Parse salesman routes from AMPL ``display x`` output.

    AMPL may print ``x`` either as individual assignments ``x[s,i,j] = 1`` or as
    matrices.  This parser supports both formats.
    """

    edges = []

    # First, look for ``x[s,i,j] = 1`` lines
    pattern = re.compile(r"x\[(\d+),(\d+),(\d+)\]\s*=\s*1")
    for s, i, j in pattern.findall(ampl_text):
        edges.append((int(s), int(i), int(j)))

    # If nothing found, attempt to parse matrix form "x [s,*,*]" printed by AMPL
    if not edges:
        lines = iter(ampl_text.splitlines())
        for line in lines:
            m = re.match(r"x\s*\[(\d+),\*,\*\]", line.strip())
            if not m:
                continue
            s = int(m.group(1))

            # Read header with column indices
            header = next(lines, "")
            cols = [int(n) for n in re.findall(r"-?\d+", header)]

            for row in lines:
                row = row.strip()
                if not row or row.startswith("[") or row.startswith(";"):
                    break
                tokens = re.findall(r"-?\d+", row)
                if len(tokens) != len(cols) + 1:
                    continue
                i = int(tokens[0])
                vals = tokens[1:]
                for col, val in zip(cols, vals):
                    if val == "1":
                        edges.append((s, i, col))

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
