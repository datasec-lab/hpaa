from datetime import datetime
import os, random, string
from tqdm import tqdm
import pandas as pd


M = ["M1", "M2", "M3", "M4", "M5", "M6"] # Spatial Placement
L = {
    "W": "Word",
    "T": "Token",
    "Mix": "Mixed"
} # Granularity
S = {
    "B": "Bold",
    "Col": "Color",
    "Hi": "Highlight",
    "Pre": "Precomposed",
    "Cap": "Cap-Flipping",
    "Cloze": "Cloze"
} # Stylistic Transformations


def gen_HPAA(args):
    gen = gen_HPAA_samples(args)
    gen.generate_adv_samples()


class gen_HPAA_samples:
    def __init__(self, args):
        # random seed
        self.seed = getattr(args, "seed", 42)
        random.seed(self.seed)

        # benign sentence b
        self.b_pool = None  # stays None for Option A/B/C; populated for Option D
        if args.benign_sentence_choice == "Given":
            self.b = args.benign
        else:
            filename = f"benign.{args.benign_sentence_choice}.csv"
            filepath = os.path.join(args.b_dataset_folder, filename)
            b_list = pd.read_csv(filepath, usecols = [args.benign])
            if args.toxic_sentence_choice == "Given":
                # Option C: sample one benign, deterministic by seed
                self.b = b_list[args.benign].sample(n=1, random_state=self.seed).item()
            else:
                # Option D: store entire pool, sample per toxic sentence
                self.b_pool = b_list[args.benign].tolist()
                self.b = None  # will be set per iteration
        
        # toxic sentence t
        if args.toxic_sentence_choice == "Given":
            self.t = [args.toxic]
        else:
            filename = f"toxic.{args.toxic_sentence_choice}.csv"
            filepath = os.path.join(args.t_dataset_folder, filename)
            t_list = pd.read_csv(filepath, usecols = [args.toxic])
            self.t = t_list[args.toxic].tolist()
        
        # configuration: m-l-s
        self.m = args.mode
        self.l = args.granularity
        self.s = args.stylistic_transformation
        self.config_str = f"{self.m}-{self.l}-{self.s}"
        
        # output
        adv_prefix = getattr(args, "adv_prefix", "adv")
        self.save_prefix = os.path.join(
            args.hpaa_folder,
            f"{adv_prefix}.{self.config_str}",
        )
        self.save_csv = self.save_prefix + ".csv"

    def _get_benign(self, idx):
        if self.b_pool is not None:
            rng = random.Random(self.seed + idx)
            return rng.choice(self.b_pool)
        return self.b

    def generate_adv_samples(self) -> None:
        results = []
        for idx, t in tqdm(list(enumerate(self.t)), total=len(self.t)):
            b = self._get_benign(idx)
            b_prepare, t_prepare = self.apply_l(b, t)
            c = self.apply_m(b_prepare, t_prepare)
            x = self.apply_s(c)
            x0 = x.replace("\n", " ")
            
            if idx == 0:
                print(f"demo of the HPAA sample:\n\n{x}")
            
            result = {}
            result["idx"] = idx
            result["b"] = b
            result["t"] = t
            result["c"] = c
            result["x"] = x
            result["x0"] = x0
            result["config"] = self.config_str
            results.append(result)
        
        results = pd.DataFrame(results)
        results.to_csv(f"{self.save_csv}", index=False)
    
    def apply_l(self, b, t):
        t_prepare = [
            self._split_word_to_fragments(word)
            for word in t.strip().split()
        ]
        b_prepare = self._break_to_lines(b, min_lines=len(t_prepare))
        return b_prepare, t_prepare
    
    def _break_to_lines(self, b, min_lines=1):
        words = b.lower().split()
        if not words:
            return [""] * max(min_lines, 1)

        # Start with default max line width and shrink until we have enough lines
        max_width = 40
        while max_width >= 1:
            b_prepare = self._wrap_words(words, max_width)
            if len(b_prepare) >= min_lines:
                return b_prepare
            max_width = max_width // 2
        
        b_prepare = [""] * min_lines
        for i, w in enumerate(words):
            line_idx = i % min_lines
            if b_prepare[line_idx]:
                b_prepare[line_idx] += " " + w
            else:
                b_prepare[line_idx] = w
        return b_prepare

    @staticmethod
    def _wrap_words(words, max_width):
        """Break a word list into lines of at most max_width characters."""
        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + (1 if current_line else 0) <= max_width:
                if current_line:
                    current_line += " "
                current_line += word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines

    def _split_word_to_fragments(self, word):
        word = word.strip('"').upper()
        length = len(word)

        if self.l == "W":
            return [word]

        elif self.l == "T":
            if length <= 3:
                return [word]
            frags = []
            pos = 0
            while length - pos > 4:
                take = 3 if (length - pos - 3) >= 2 else 2
                frags.append(word[pos : pos + take])
                pos += take
            rem = length - pos
            if rem == 4:
                frags.append(word[pos : pos + 2])
                frags.append(word[pos + 2 : pos + 4])
            else:
                frags.append(word[pos : pos + rem])
            return frags

        elif self.l == "Mix":
            fragments = []
            pos = 0
            while pos < length:
                remaining = length - pos
                if remaining > 4:
                    frag_len = min(3, remaining)
                else:
                    frag_len = min(2, remaining)
                fragment = word[pos : pos + frag_len]
                fragments.append(fragment)
                pos += frag_len
            return fragments
        
        return fragments

    def apply_m(self, b_prepare, t_prepare):
        if self.m == "M6":
            if not b_prepare:
                return "\n".join(b_prepare)

            if self.l == "W":
                inject_units = [frag[0].upper() for frag in t_prepare if frag and frag[0]]
                
                for idx, virus_word in enumerate(inject_units):
                    line_idx = idx % len(b_prepare)
                    words = b_prepare[line_idx].split()
                    insert_pos = random.randrange(len(words) + 1)
                    words[insert_pos:insert_pos] = [virus_word]
                    b_prepare[line_idx] = " ".join(words)
                return "\n".join(b_prepare)

            else:
                for idx, frags in enumerate(t_prepare):
                    if not frags:
                        continue
                    frags = [p.upper() for p in frags if p]
                    if not frags:
                        continue

                    line_idx = idx % len(b_prepare)
                    words = b_prepare[line_idx].split()
                    if not words:
                        b_prepare[line_idx] = " ".join(frags)
                        continue

                    insert_pos = random.randrange(len(words) + 1)

                    new_words = []
                    new_words.extend(words[:insert_pos])

                    k = 0
                    for j, frag in enumerate(frags):
                        new_words.append(frag)
                        if insert_pos + k < len(words):
                            new_words.append(words[insert_pos + k])
                            k += 1

                    new_words.extend(words[insert_pos + k :])
                    b_prepare[line_idx] = " ".join(new_words)

                return "\n".join(b_prepare)

        for row_idx, fragments in enumerate(t_prepare):
            if row_idx >= len(b_prepare):
                break

            words = b_prepare[row_idx].split()
            if len(words) < 1:
                continue

            if self.m == "M1":
                new_words = []
                for j, fragment in enumerate(fragments):
                    new_words.append(fragment)
                    if j < len(words):
                        new_words.append(words[j])
                new_words.extend(words[len(fragments) :])
                b_prepare[row_idx] = " ".join(new_words)

            elif self.m == "M2":
                new_words = []
                mid = len(words) // 2 - len(fragments) // 2
                mid = max(0, mid)
                new_words.extend(words[:mid])
                for j, fragment in enumerate(fragments):
                    new_words.append(fragment)
                    if mid + j < len(words):
                        new_words.append(words[mid + j])
                new_words.extend(words[mid + len(fragments) :])
                b_prepare[row_idx] = " ".join(new_words)

            elif self.m == "M3":
                new_words = []
                num_fragments = len(fragments)
                num_words = len(words)
                insert_start = max(0, num_words - num_fragments)
                new_words.extend(words[:insert_start])

                for j in range(num_fragments - 1):
                    new_words.append(fragments[j])
                    if insert_start + j < num_words:
                        new_words.append(words[insert_start + j])

                if num_fragments > 0:
                    if insert_start + num_fragments - 1 < num_words:
                        new_words.append(words[insert_start + num_fragments - 1])
                    new_words.append(fragments[-1])
                b_prepare[row_idx] = " ".join(new_words)

            elif self.m == "M4":
                new_words = words.copy()
                frags = fragments

                if not frags:
                    b_prepare[row_idx] = " ".join(new_words)
                else:
                    insert_pos = min(row_idx, len(new_words))

                    left = new_words[:insert_pos]
                    right = new_words[insert_pos:]
                    merged = []
                    k = 0
                    for f in frags:
                        merged.append(f)
                        if k < len(right):
                            merged.append(right[k])
                            k += 1
                    merged.extend(right[k:])
                    new_words = left + merged

                    b_prepare[row_idx] = " ".join(new_words)

            elif self.m == "M5":
                new_words = words.copy()
                frags = fragments

                if not frags:
                    b_prepare[row_idx] = " ".join(new_words)
                else:
                    insert_pos = max(0, len(new_words) - (row_idx + 1))

                    left = new_words[:insert_pos]
                    right = new_words[insert_pos:]

                    merged = []
                    k = 0
                    for f in frags:
                        merged.append(f)
                        if k < len(right):
                            merged.append(right[k])
                            k += 1
                    merged.extend(right[k:])
                    new_words = left + merged

                    b_prepare[row_idx] = " ".join(new_words)
            else:
                continue

        return "\n".join(b_prepare)

    def apply_s(self, text):
        if self.s == "Cap":
            return text
        else:
            loc_char = self._find_loc(text)
            text = list(text)
            for loc in loc_char:
                char = loc_char[loc]
                if self.s == "Pre":
                    text[loc] = self._format_s_precomposed(char)
                elif self.s == "B":
                    text[loc] = self._format_s_bold(char)
                elif self.s == "Col":
                    text[loc] = self._format_s_color(char)
                elif self.s == "Hi":
                    text[loc] = self._format_s_highlight(char)
                elif self.s == "Cloze":
                    text[loc] = "_"
            return "".join(text)

    def _find_loc(self, text):
        loc_char = {}
        for i, char in enumerate(list(text)):
            if char.isupper():
                loc_char.update({i: char})
        return loc_char
    
    @staticmethod
    def _format_s_bold(c):
        if c.isupper():
            c = c.lower()
        return f"\033[1m{c}\033[0m"

    @staticmethod
    def _format_s_color(c, color="blue"):
        if c.isupper():
            c = c.lower()
        return f"\033[34m{c}\033[0m"

    @staticmethod
    def _format_s_highlight(c):
        if c.isupper():
            c = c.lower()
        return f"\033[30;43m{c}\033[0m"

    @staticmethod
    def _format_s_precomposed(c):
        if c.isupper():
            c = c.lower()
        precomposed_letters = {
            **{c: chr(0x24B6 + i) for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")},
            **{c: chr(0x24D0 + i) for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")},
        }
        return precomposed_letters.get(c, c)

    @staticmethod
    def _generate_random_string(length=1000, ratio=0.9):
        good_chars = string.ascii_lowercase + " "
        bad_chars = ",.! "  # string.punctuation: '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

        result = []
        for _ in range(length):
            if random.random() < ratio:
                result.append(random.choice(good_chars))
            else:
                result.append(random.choice(bad_chars))

        return "".join(result)