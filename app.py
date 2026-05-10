import math
import heapq
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ── City graph with coordinates (lat, lon) and connections ──────────────────
CITIES = {
    "Karachi":   {"lat": 24.86, "lon": 67.01},
    "Lahore":    {"lat": 31.55, "lon": 74.34},
    "Islamabad": {"lat": 33.72, "lon": 73.06},
    "Peshawar":  {"lat": 34.01, "lon": 71.57},
    "Quetta":    {"lat": 30.19, "lon": 67.00},
    "Multan":    {"lat": 30.20, "lon": 71.47},
    "Faisalabad":{"lat": 31.42, "lon": 72.99},
    "Hyderabad": {"lat": 25.37, "lon": 68.37},
    "Sukkur":    {"lat": 27.70, "lon": 68.86},
    "Rawalpindi":{"lat": 33.60, "lon": 73.04},
}

# Undirected graph edges (city_a, city_b, road_km)
EDGES = [
    ("Karachi",    "Hyderabad",  165),
    ("Hyderabad",  "Sukkur",     400),
    ("Sukkur",     "Quetta",     550),
    ("Sukkur",     "Multan",     460),
    ("Multan",     "Lahore",     340),
    ("Multan",     "Faisalabad", 260),
    ("Lahore",     "Faisalabad", 130),
    ("Lahore",     "Rawalpindi", 280),
    ("Rawalpindi", "Islamabad",   15),
    ("Islamabad",  "Peshawar",   170),
    ("Rawalpindi", "Peshawar",   175),
    ("Quetta",     "Karachi",    700),
    ("Faisalabad", "Rawalpindi", 270),
]

def build_graph():
    graph = {city: [] for city in CITIES}
    for a, b, dist in EDGES:
        graph[a].append((b, dist))
        graph[b].append((a, dist))
    return graph

def haversine(city_a, city_b):
    """Straight-line heuristic distance in km."""
    R = 6371
    lat1, lon1 = math.radians(CITIES[city_a]["lat"]), math.radians(CITIES[city_a]["lon"])
    lat2, lon2 = math.radians(CITIES[city_b]["lat"]), math.radians(CITIES[city_b]["lon"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def greedy_best_first_search(start, goal):
    """
    Greedy Best-First Search using heuristic h(n) = haversine distance to goal.
    Priority queue ordered purely by h(n) — no path cost considered.
    Returns: path list, total road distance, nodes explored, steps log.
    """
    graph = build_graph()
    # heap: (heuristic, city, path_so_far, road_dist_so_far)
    heap = [(haversine(start, goal), start, [start], 0)]
    visited = set()
    steps = []

    while heap:
        h, current, path, road_dist = heapq.heappop(heap)

        if current in visited:
            continue
        visited.add(current)

        steps.append({
            "city": current,
            "heuristic_km": round(h, 1),
            "road_dist_so_far": round(road_dist, 1),
            "path": list(path),
        })

        if current == goal:
            return {
                "found": True,
                "path": path,
                "total_road_km": round(road_dist, 1),
                "nodes_explored": len(visited),
                "steps": steps,
                "straight_line_km": round(haversine(start, goal), 1),
            }

        for neighbor, edge_dist in graph[current]:
            if neighbor not in visited:
                h_n = haversine(neighbor, goal)
                heapq.heappush(heap, (h_n, neighbor, path + [neighbor], road_dist + edge_dist))

    return {"found": False, "path": [], "steps": steps}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", cities=sorted(CITIES.keys()))

@app.route("/api/cities")
def get_cities():
    return jsonify({"cities": sorted(CITIES.keys()), "coords": CITIES, "edges": EDGES})

@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json()
    start, goal = data.get("start"), data.get("goal")
    if not start or not goal:
        return jsonify({"error": "start and goal required"}), 400
    if start == goal:
        return jsonify({"error": "Start and goal must differ"}), 400
    if start not in CITIES or goal not in CITIES:
        return jsonify({"error": "Unknown city"}), 400
    result = greedy_best_first_search(start, goal)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5050)
