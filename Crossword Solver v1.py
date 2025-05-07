# === IMPORTS === #
import sqlite3
import re
import pandas as pd
import numpy as np
from itertools import product
from openai import OpenAI
import unicodedata
from collections import defaultdict

# === FILE PATHS === # update these before running
api_key_location = r""
scowl_location = r""
database_location = r""

# === INITIALIZE STABILITY TRACKING === #
locked_cells = set()                      # Cells that shouldn't change again
cell_stability = defaultdict(int)         # (i,j) → # times cell unchanged

# === CONNECT TO DATABASE === #
connection = sqlite3.connect(database_location)
Puzzle_ID = 15501

# === LOAD CLUES === #
clues_data = pd.read_sql_query(f"SELECT acrosstext, downtext FROM text WHERE id = {Puzzle_ID}", connection)
across_clues = clues_data.at[0, "acrosstext"]
down_clues = clues_data.at[0, "downtext"]

# === LOAD GRID === #
grid = pd.read_sql_query(f"SELECT grid FROM grids WHERE id = {Puzzle_ID}", connection).iloc[0]["grid"]
grid = np.frombuffer(grid, dtype=np.uint8) - 48
grid = grid.reshape(13, 13)
grid_df = pd.DataFrame(grid.astype(str))  # Use strings so we can place letters
# Convert grid 1s to blanks (underscores), preserve 0s as black squares
for i in range(13):
    for j in range(13):
        if grid[i, j] == 1:
            grid_df.iat[i, j] = "_"
        else:
            grid_df.iat[i, j] = "#"  # optional: mark black squares visually


connection.close()

# === EXTRACT CLUE DATA === #
pattern = re.compile(r'(\d+(?:,\d+(?:across|down)?)*)\s((?:See\s\d+(?:\s(?:across|down))?)|(?:.+?(?=\s\(\d+(?:[,-]\d+)*\))))\s?(?:\((\d+(?:[,-]\d+)*)\))?')
across_clues_list = pattern.findall(across_clues)
down_clues_list = pattern.findall(down_clues)

# === CREATE clues_df === #
df_across = pd.DataFrame(across_clues_list, columns=["clue_number", "hint", "length"])
df_down = pd.DataFrame(down_clues_list, columns=["clue_number", "hint", "length"])
df_across["direction"] = "A"
df_down["direction"] = "D"
clues_df = pd.concat([df_across, df_down], ignore_index=True)
clues_df["clue_number_refined"] = clues_df["clue_number"].str.split(",").str[0]
clues_df["clue_id"] = clues_df["clue_number_refined"] + clues_df["direction"]

# === DETERMINE COORDS FOR EACH CLUE === #
solution_coordinates = []
clue_id_grid = np.zeros_like(grid, dtype=int)
count = 1

for i in range(13):
    for j in range(13):
        if grid[i, j] == 0:
            continue
        is_start_across = (j == 0 or grid[i, j-1] == 0) and (j + 1 < 13 and grid[i, j+1] == 1)
        is_start_down = (i == 0 or grid[i-1, j] == 0) and (i + 1 < 13 and grid[i+1, j] == 1)

        if is_start_across or is_start_down:
            clue_id_grid[i, j] = count

            if is_start_across:
                coords = [(i, y) for y in range(j, 13) if grid[i, y] == 1]
                solution_coordinates.append((count, "A", coords))

            if is_start_down:
                coords = [(x, j) for x in range(i, 13) if grid[x, j] == 1]
                solution_coordinates.append((count, "D", coords))

            count += 1

solution_coordinates_df = pd.DataFrame(solution_coordinates, columns=["clue_number", "direction", "coords"])
solution_coordinates_df["clue_id"] = solution_coordinates_df["clue_number"].astype(str) + solution_coordinates_df["direction"]

# === MERGE AND BUILD MAIN PUZZLE DF === #
puzzle_df = pd.merge(clues_df, solution_coordinates_df, on="clue_id", how="inner")
puzzle_df.set_index("clue_id", inplace=True)
puzzle_df = puzzle_df[puzzle_df["length"].str.strip() != ""]

# Extract any secondary clue references
puzzle_df["extra_clues"] = clues_df["clue_number"].apply(
    lambda x: re.findall(r"\d+\w?", str(x))[1:] if pd.notnull(x) else []
)
puzzle_df["extra_clues"] = puzzle_df["extra_clues"].apply(
    lambda lst: [s.upper() for s in lst] if isinstance(lst, list) else []
)

# Expand coords for multi-part solutions
for clue_id in puzzle_df.index:
    for extra_id in puzzle_df.at[clue_id, "extra_clues"]:
        if re.search(r"[AD]", extra_id):
            puzzle_df.at[clue_id, "coords"].extend(puzzle_df.loc[extra_id, "coords"])
        else:
            matches = puzzle_df[puzzle_df["clue_number"] == extra_id]
            if not matches.empty:
                puzzle_df.at[clue_id, "coords"].extend(matches.iloc[0]["coords"])

# Add anagram metadata
puzzle_df["anagram_flag"] = puzzle_df["hint"].str.contains(r"\(anag(?:\.|ram)?\)", case=False)
puzzle_df["anagram_part"] = None
puzzle_df["non_anagram_part"] = None

for clue_id in puzzle_df.index:
    if puzzle_df.at[clue_id, "anagram_flag"]:
        split_hint = re.split(r"\s?[-–—]\s?", puzzle_df.at[clue_id, "hint"])
        if len(split_hint) == 1:
            match = re.findall(r"(.*?)(?=\s\(anag(?:\.|ram)?\))", split_hint[0])
            if match:
                puzzle_df.at[clue_id, "anagram_part"] = match[0]
        else:
            if re.search(r"\(anag(?:\.|ram)?\)", split_hint[0]):
                match = re.findall(r"(.*?)(?=\s\(anag(?:\.|ram)?\))", split_hint[0])
                if match:
                    puzzle_df.at[clue_id, "anagram_part"] = match[0]
                puzzle_df.at[clue_id, "non_anagram_part"] = split_hint[1]
            else:
                match = re.findall(r"(.*?)(?=\s\(anag(?:\.|ram)?\))", split_hint[1])
                if match:
                    puzzle_df.at[clue_id, "anagram_part"] = match[0]
                puzzle_df.at[clue_id, "non_anagram_part"] = split_hint[0]

# Add intersector metadata
puzzle_df["intersectors"] = None
for clue_id in puzzle_df.index:
    puzzle_df.at[clue_id, "intersectors"] = []
    for other_id in puzzle_df.index:
        if clue_id != other_id:
            if any(coord in puzzle_df.at[clue_id, "coords"] for coord in puzzle_df.at[other_id, "coords"]):
                puzzle_df.at[clue_id, "intersectors"].append(other_id)

# Add blank solution templates
puzzle_df["solution_so_far"] = puzzle_df["length"].apply(
    lambda l: "/".join(["_" * int(x) for x in re.split(r"[-,]", l)])
)
puzzle_df["length_as_list"] = puzzle_df["length"].apply(lambda x: list(map(int, re.split(r"[-,]", x))))

# Final setup columns
puzzle_df["potential_solutions"] = [[] for _ in range(len(puzzle_df))]
puzzle_df["fitting_solutions"] = None
puzzle_df["probable_solution"] = None
puzzle_df["best_fit"] = None
puzzle_df["factual_solution"] = None

# === LOAD SCOWL WORDLIST === #
word_set = set()
word_pattern = re.compile(r'^[a-zA-Z]+$')
with open(scowl_location, encoding="utf-8") as f:
    for line in f:
        for word in line.split():
            clean = re.sub(r'[^\w\s]', '', word).upper()
            if word_pattern.fullmatch(clean):
                word_set.add(clean)

# === INIT OPENAI CLIENT === #
with open(api_key_location) as f:
    api_key = f.read()
client = OpenAI(api_key=api_key)
model = "gpt-4o-mini"
temperature = 0.2

# === SOLVER UTILITIES === #

# === GPT Query Function ===
def query_GPT(hint, solution_so_far, length):
    prompt = f"""
    Crossword clue: "{hint}"
    Known letters: "{solution_so_far}"
    Solution length: {length}

    Return a numbered list of potential solutions only. If more than one word, use forward slashes between words. No explanation.
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )

        # Raw output from GPT
        content = response.choices[0].message.content

        # Debug print (optional)
        print(f"\n📨 GPT raw response:\n{content}\n")

        # Extract numbered lines
        matches = re.findall(r"^\d+\.\s*(.+)", content, re.MULTILINE)

        if not matches:
            print("No matches found in GPT response.")
            return []

        cleaned_solutions = []
        for sol in matches:
            # Normalize & clean
            sol = unicodedata.normalize("NFD", sol.upper())
            sol = "".join(c for c in sol if unicodedata.category(c) != "Mn" or c == "/")
            sol = re.sub(r"[^\w/]", "", sol)
            sol = re.sub(r"[\s\-]+", "/", sol)

            # Validate format and length
            parts = sol.split("/")
            if len(parts) == len(length) and all(len(part) == length[i] for i, part in enumerate(parts)):
                cleaned_solutions.append("/".join(parts))

        return list(set(cleaned_solutions))

    except Exception as e:
        print(f"Error in query_GPT: {e}")
        return []


# === GPT Fit Ranking Function ===
def query_GPT_fit(hint, candidates):
    formatted = ", ".join(candidates)
    prompt = f"""
Crossword clue: "{hint}"

The following are potential solutions: {formatted}

Please rank them by likelihood of being correct. Return only a numbered list with the exact solutions provided.
If a solution contains multiple words, they are separated by a slash.
You must only rank from the given solutions. Do not split, reword, or reinterpret them.

Return no commentary or explanation — just the ranked list.
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )

        content = response.choices[0].message.content
        print(f"\n📨 GPT FIT response:\n{content.strip()}\n")

        # Extract just the ranked items from the list
        ranked_lines = re.findall(r"^\d+\.\s*(.+)", content, re.MULTILINE)

        # Clean and match exactly against candidate list
        ranked_cleaned = []
        for line in ranked_lines:
            cleaned = unicodedata.normalize("NFD", line.upper())
            cleaned = re.sub(r"[^\w/\s\-]", "", cleaned)
            cleaned = re.sub(r"\s+", "/", cleaned).strip("/")
            ranked_cleaned.append(cleaned)

        # Return the first match in order that exists in candidates
        for line in ranked_cleaned:
            if line in [c.upper() for c in candidates]:
                return line

        # Fallback
        print("No exact match found — using fallback")
        return candidates[0]

    except Exception as e:
        print(f"GPT fit error: {e}")
        return candidates[0]
    

# === Find Anagrams ===
def is_match(solution_part, word):
    return len(solution_part) == len(word) and all(
        s == "_" or s == w for s, w in zip(solution_part, word)
    )

def backtrack_anagram(letters, sol_parts, length, idx, current, results):
    if idx == len(length):
        results.add("/".join(current))
        return
    part = sol_parts[idx]
    for word in word_set:
        if len(word) == length[idx] and is_match(part, word):
            temp = list(letters)
            try:
                for c in word:
                    temp.remove(c)
                backtrack_anagram(temp, sol_parts, length, idx + 1, current + [word], results)
            except:
                continue

def find_anagrams(hint, solution_so_far, length):
    results = set()
    clean = hint.upper().replace(" ", "")
    parts = solution_so_far.split("/")
    backtrack_anagram(list(clean), parts, length, 0, [], results)
    return sorted(results)


# === Find Words from Word List ===
def find_words(partial_solution, length):
    patterns = partial_solution.split('/')
    if len(patterns) != len(length):
        raise ValueError("Mismatch between parts and lengths")

    word_options = []
    for i, pattern in enumerate(patterns):
        regex = re.compile('^' + pattern.replace('_', '.') + '$')
        word_options.append([w for w in word_set if regex.fullmatch(w)])

    return ['/'.join(words) for words in product(*word_options)]


# === Strategy Functions ===
def standard_clue(clue_id):
    sols = query_GPT(puzzle_df.at[clue_id, "hint"], puzzle_df.at[clue_id, "solution_so_far"], puzzle_df.at[clue_id, "length_as_list"])
    puzzle_df.at[clue_id, "potential_solutions"] += sols
    puzzle_df.at[clue_id, "potential_solutions"] = list(set(puzzle_df.at[clue_id, "potential_solutions"]))

def pure_anagram(clue_id):
    sols = find_anagrams(puzzle_df.at[clue_id, "anagram_part"], puzzle_df.at[clue_id, "solution_so_far"], puzzle_df.at[clue_id, "length_as_list"])
    puzzle_df.at[clue_id, "potential_solutions"] += sols
    puzzle_df.at[clue_id, "potential_solutions"] = list(set(puzzle_df.at[clue_id, "potential_solutions"]))

def standard_anagram(clue_id):
    ana = find_anagrams(puzzle_df.at[clue_id, "anagram_part"], puzzle_df.at[clue_id, "solution_so_far"], puzzle_df.at[clue_id, "length_as_list"])
    gpt = query_GPT(puzzle_df.at[clue_id, "non_anagram_part"], puzzle_df.at[clue_id, "solution_so_far"], puzzle_df.at[clue_id, "length_as_list"])
    common = list(set(ana).intersection(gpt)) or ana
    puzzle_df.at[clue_id, "potential_solutions"] += common
    puzzle_df.at[clue_id, "potential_solutions"] = list(set(puzzle_df.at[clue_id, "potential_solutions"]))


# === Grid → solution_so_far Update ===
def update_solution_so_far_from_grid():
    for clue_id in puzzle_df.index:
        coords = puzzle_df.at[clue_id, "coords"]
        lengths = puzzle_df.at[clue_id, "length_as_list"]
        letters = [grid_df.iat[i, j] for (i, j) in coords]

        words = []
        idx = 0
        for l in lengths:
            word = ''.join(letters[idx:idx+l])
            word = re.sub(r"[^A-Z_]", "_", word.upper())
            words.append(word)
            idx += l

        puzzle_df.at[clue_id, "solution_so_far"] = '/'.join(words)