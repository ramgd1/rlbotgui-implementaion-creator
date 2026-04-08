# BOT SETUP

An interactive CLI tool that scaffolds RLBot (Rocket League) bot folders from pre-trained neural network weights.

**by q7cj(ram)**

---

## What it does

`main.py` asks you a few questions about your bot and generates a ready-to-run RLBot bot folder containing:

| File | Description |
|---|---|
| `agent.py` | Loads `SHARED_HEAD.LT` + `POLICY.LT` weights and exposes an `act()` method |
| `bot.py` | Full `BaseAgent` subclass wired to the agent and obs builder |
| `bot.cfg` | RLBot config pointing at `bot.py` |
| `obs.py` | Observation builder (copied from assets) |
| `discrete_policy.py` | `DiscreteFF` policy network definition |
| `your_act.py` | `LookupAction` action parser |
| `util/` | Shared game-state utilities |
| `appearance.cfg` | Bot appearance config |
| `requirements.txt` | Bot-level dependencies |

After generation you drop your own `SHARED_HEAD.LT` and `POLICY.LT` weight files into the output folder and the bot is ready to run.

---

## Requirements

`main.py` itself uses only the Python standard library — no pip install needed to run the generator.

The **generated bot** requires:

```
rlbot==1.*
torch          # CPU build
numpy<2.0
```

---

## Usage

```bash
python main.py
```

You will be prompted for:

1. **Bot name** — used as the output folder name and Python class name
2. **Obs type**
   - `1` → `AdvancedObs` (standard 1v1 / 2v2 / 3v3)
   - `2` → `AdvancedObsPadded` (variable team sizes / MultiMode)
3. **Obs size** — must match the size your network was trained on (common values: `109`, `167`, `225`)
4. **Shared layer sizes** — comma-separated integers, e.g. `1024,1024,512`
5. **Policy layer sizes** — comma-separated integers, e.g. `512,256`

### Example session

```
> name              my_bot
> select            1        (AdvancedObs)
> size              109
> shared layers     1024,1024,512
> policy layers     512,256
```

Output folder `my_bot/` is created next to `main.py`. Drop `SHARED_HEAD.LT` and `POLICY.LT` into it, then load it in RLBot.

---

## Project structure

```
.
├── main.py            # Generator script
├── requirements.txt   # Bot dependency list (copied into generated folders)
└── assets/            # Static files copied into every generated bot
    ├── discrete_policy.py
    ├── your_act.py
    ├── obs.py
    ├── appearance.cfg
    ├── requirements.txt
    └── util/
```

---

## Network architecture

The generator wires up a `DiscreteFF` policy network with two sub-modules:

- **shared_head** — loaded from `SHARED_HEAD.LT`
- **policy** — loaded from `POLICY.LT`

Both are loaded onto CPU with `torch.set_num_threads(1)` for deterministic single-threaded inference inside RLBot.
