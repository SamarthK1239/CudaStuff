#!/usr/bin/env python3
"""
1-Billion-Row Challenge — Data Generator

Generates a measurements.txt file in the canonical format:
    StationName;temperature
where temperature has exactly one decimal place.

Usage:
    python main.py                          # 1 billion rows → measurements.txt
    python main.py --rows 1000000 -o test.txt  # 1M rows → test.txt
"""

import argparse
import os
import random
import sys
import time

# 413 real-world weather station names (canonical 1BRC set)
STATIONS = [
    "Abha", "Abidjan", "Abéché", "Accra", "Addis Ababa", "Adelaide", "Aden",
    "Ahvaz", "Albuquerque", "Alexandra", "Alexandria", "Algiers", "Alice Springs",
    "Almaty", "Amsterdam", "Anadyr", "Anchorage", "Andorra la Vella", "Ankara",
    "Antananarivo", "Antsiranana", "Arkhangelsk", "Ashgabat", "Asmara", "Assab",
    "Astana", "Athens", "Atlanta", "Auckland", "Austin", "Baghdad", "Baku",
    "Baltimore", "Bamako", "Bangkok", "Bangui", "Banjul", "Barcelona",
    "Bata", "Batumi", "Beijing", "Beirut", "Belgrade", "Belize City", "Benghazi",
    "Bergen", "Berlin", "Bilbao", "Birao", "Bishkek", "Bissau", "Blantyre",
    "Bloemfontein", "Boise", "Bordeaux", "Bosaso", "Boston", "Bouaké",
    "Bratislava", "Brazzaville", "Bridgetown", "Brisbane", "Brussels",
    "Bucharest", "Budapest", "Bujumbura", "Bulawayo", "Bursa", "Busan",
    "Cabo San Lucas", "Cairns", "Cairo", "Calgary", "Canberra", "Cape Town",
    "Casablanca", "Cayenne", "Charlotte", "Chiang Mai", "Chicago",
    "Chihuahua", "Chittagong", "Chișinău", "Christchurch", "City of San Marino",
    "Colombo", "Columbus", "Conakry", "Copenhagen", "Cotonou", "Cracow",
    "Da Lat", "Da Nang", "Dakar", "Dallas", "Damascus", "Dampier",
    "Dar es Salaam", "Darwin", "Denpasar", "Denver", "Detroit", "Dhaka",
    "Dikson", "Dili", "Djibouti", "Dodoma", "Dolisie", "Douala", "Dubai",
    "Dublin", "Dunedin", "Durban", "Dushanbe", "Edinburgh", "Edmonton",
    "El Paso", "Entebbe", "Erbil", "Erzurum", "Fairbanks", "Fianarantsoa",
    "Flores,  Petén", "Frankfurt", "Freetown", "Fresno", "Fukuoka", "Gaborone",
    "Gabès", "Gangtok", "Garissa", "Garoua", "George Town", "Ghanzi",
    "Gjoa Haven", "Guadalajara", "Guangzhou", "Guatemala City", "Halifax",
    "Hamburg", "Hamilton", "Hanoi", "Harare", "Harbin", "Hargeisa", "Hat Yai",
    "Havana", "Helsinki", "Heraklion", "Hiroshima", "Ho Chi Minh City", "Hobart",
    "Hong Kong", "Honiara", "Honolulu", "Houston", "Ifrane", "Indianapolis",
    "Indore", "Iqaluit", "Irkutsk", "Istanbul", "Jacksonville",
    "Jakarta", "Jayapura", "Jerusalem", "Johannesburg", "Jos", "Juba",
    "Kabul", "Kampala", "Kandi", "Kankan", "Kano", "Kansas City", "Karachi",
    "Karonga", "Kathmandu", "Khartoum", "Kingston", "Kinshasa", "Kolkata",
    "Kuala Lumpur", "Kumasi", "Kunming", "Kuopio", "Kuwait City", "Kyiv",
    "Kyoto", "La Ceiba", "La Paz", "Lagos", "Lahore", "Lake Havasu City",
    "Lake Tekapo", "Las Palmas de Gran Canaria", "Las Vegas", "Launceston",
    "Lhasa", "Libreville", "Lisbon", "Livingstone", "Ljubljana", "Lomé",
    "London", "Los Angeles", "Louisville", "Luanda", "Lubumbashi", "Lusaka",
    "Luxembourg City", "Lviv", "Lyon", "Madrid", "Mahajanga", "Makassar",
    "Makurdi", "Malabo", "Malé", "Managua", "Manama", "Mandalay", "Mangalore",
    "Manila", "Maputo", "Marrakesh", "Marseille", "Maun", "Medan",
    "Medellín", "Melbourne", "Memphis", "Mexicali", "Mexico City", "Miami",
    "Milan", "Milwaukee", "Minneapolis", "Minsk", "Mogadishu", "Mombasa",
    "Monaco", "Moncton", "Monterrey", "Montreal", "Moscow", "Mumbai", "Murmansk",
    "Muscat", "Mzuzu", "N'Djamena", "Naha", "Nairobi", "Nakhon Ratchasima",
    "Napier", "Napoli", "Nashville", "Nassau", "Ndola", "New Delhi",
    "New Orleans", "New York City", "Ngaoundéré", "Niamey", "Nicosia",
    "Norilsk", "Nouakchott", "Novosibirsk", "Nuuk", "Odesa", "Odienné",
    "Oklahoma City", "Omaha", "Oranjestad", "Oslo", "Ottawa", "Ouagadougou",
    "Ouarzazate", "Oulu", "Palembang", "Palermo", "Palm Springs", "Palmerston North",
    "Panama City", "Paramaribo", "Paris", "Perth", "Petropavlovsk-Kamchatsky",
    "Philadelphia", "Phnom Penh", "Phoenix", "Pittsburgh", "Podgorica",
    "Pointe-Noire", "Pontianak", "Port Moresby", "Port Sudan", "Port Vila",
    "Port-Gentil", "Portland (OR)", "Porto", "Porto Alegre", "Prague",
    "Pretoria", "Pyongyang", "Québec", "Quito", "Rabat", "Rangpur",
    "Rapid City", "Rawalpindi", "Reggane", "Reykjavik", "Riga", "Riyadh",
    "Rome", "Roseau", "Rostov-on-Don", "Sacramento", "Saint Petersburg",
    "Salt Lake City", "San Antonio", "San Diego", "San Francisco",
    "San José", "San Juan", "San Salvador", "Sana'a", "Santiago",
    "Santo Domingo", "São Paulo", "Sarajevo", "Saskatoon", "Seattle",
    "Seoul", "Seville", "Shanghai", "Singapore", "Skopje", "Sochi",
    "Sofia", "Sokoto", "Split", "St. John's", "St. Louis", "Stockholm",
    "Surabaya", "Suva", "Suwałki", "Sydney", "Tabora", "Tabriz",
    "Taipei", "Tallinn", "Tamale", "Tamanrasset", "Tampa", "Tashkent",
    "Tbilisi", "Tegucigalpa", "Tehran", "Tel Aviv", "Thessaloniki",
    "Thiès", "Tijuana", "Timbuktu", "Tirana", "Toamasina", "Tokyo",
    "Toliara", "Toluca", "Toronto", "Tripoli", "Tromsø", "Tucson",
    "Tunis", "Ulaanbaatar", "Upington", "Ürümqi", "Vaduz", "Valencia",
    "Valletta", "Vancouver", "Veracruz", "Vienna", "Vientiane", "Vilnius",
    "Virginia Beach", "Vladivostok", "Warsaw", "Washington, D.C.",
    "Wau", "Wellington", "Whitehorse", "Wichita", "Willemstad",
    "Winnipeg", "Wrocław", "Xi'an", "Yakutsk", "Yangon", "Yaoundé",
    "Yellowknife", "Yerevan", "Yinchuan", "Zagreb", "Zanzibar City",
    "Zürich",
]

# Each station has a "mean" temperature; actual readings are sampled
# as mean + uniform(-15, +15), clamped to [-99.9, 99.9].
# Means are pre-assigned per station for reproducibility.
def _build_station_means(seed=42):
    rng = random.Random(seed)
    return {s: round(rng.uniform(-30.0, 45.0), 1) for s in STATIONS}


def generate(rows: int, output: str):
    station_means = _build_station_means()
    station_list = list(station_means.keys())
    n_stations = len(station_list)

    rng = random.Random(12345)
    buf_size = 64 * 1024 * 1024  # 64 MB write buffer
    start = time.time()
    written = 0

    with open(output, "w", buffering=buf_size) as f:
        # Build lines in batches for speed
        batch = 100_000
        while written < rows:
            chunk_size = min(batch, rows - written)
            lines = []
            for _ in range(chunk_size):
                idx = rng.randint(0, n_stations - 1)
                station = station_list[idx]
                mean = station_means[station]
                temp = round(mean + rng.uniform(-15.0, 15.0), 1)
                temp = max(-99.9, min(99.9, temp))
                lines.append(f"{station};{temp:.1f}\n")
            f.write("".join(lines))
            written += chunk_size

            if written % 100_000_000 == 0:
                elapsed = time.time() - start
                print(
                    f"  {written:>14,} / {rows:,} rows  "
                    f"({written * 100.0 / rows:5.1f}%)  "
                    f"[{elapsed:.1f}s]",
                    file=sys.stderr,
                )

    elapsed = time.time() - start
    size_gb = os.path.getsize(output) / (1024 ** 3)
    print(
        f"\nDone: {rows:,} rows written to {output} "
        f"({size_gb:.2f} GB) in {elapsed:.1f}s",
        file=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate measurements.txt for the 1-Billion-Row Challenge"
    )
    parser.add_argument(
        "--rows", "-n",
        type=int,
        default=1_000_000_000,
        help="Number of rows to generate (default: 1,000,000,000)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="measurements.txt",
        help="Output file path (default: measurements.txt)",
    )
    args = parser.parse_args()

    print(
        f"Generating {args.rows:,} rows → {args.output} "
        f"({len(STATIONS)} stations)",
        file=sys.stderr,
    )
    generate(args.rows, args.output)


if __name__ == "__main__":
    main()
