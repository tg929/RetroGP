
# gp_retro_repr — Problem Representation (Decision Program) for GP Retrosynthesis

This package provides the **first module** of a GP-based retrosynthesis framework:
a strongly-typed representation of steps, routes, and a minimal **Decision Program** instruction set
with an interpreter that executes programs to produce routes in the *sequential format*.

## What is included
- `Molecule`, `Inventory` (stock), `ReactionTemplate`, and registries
- `RetrosynthesisStep` and `Route` (with JSON (de)serialization)
- `Instruction` set: `Select`, `ApplyTemplate`, `Stop`
- `Program.execute()` builds a `Route` by iteratively applying retro-templates

## Three-level evaluation hooks
The module also includes molecule-level, reaction-level, and route-level checks
(validity, existence/applicability, connectivity), matching the structure used in
LLM-Syn-Planner for step validation and feedback.

## Dependencies
- RDKit (mandatory for parsing molecules)
- rdchiral (mandatory for applying **retro** templates on products)

These are intentionally *not* installed here. Install them in your environment,
then import and run.

## Example (pseudo-code)

```python
from gp_retro_repr import Inventory, ReactionTemplateRegistry, ReactionTemplate
from gp_retro_repr import Program, Select, ApplyTemplate, Stop, ExecutionConfig

# Build inventory
stock = Inventory(["CCO", "O=C=O"])  # purchasable examples

# Build template registry
reg = ReactionTemplateRegistry()
# Example retro template: [Product]>>[Reactants], ID must be unique.
reg.add(ReactionTemplate("T1", "[C:1]-[O:2]>>[C:1]=O.[O:2]"))

prog = Program([Select(0), ApplyTemplate("T1", rational="disconnection"), Stop()])
cfg = ExecutionConfig(template_registry=reg, inventory=stock)

route = prog.execute(target_smiles="CCO", config=cfg)
print(route.to_json())
```

## Next modules (not in this package)
- Search (GP/EA operators on programs, selection, diversity)
- Partial rewards & SCScore integration
- RAG/reaction mapping
```
