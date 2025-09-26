# Data Naming Convention

The `dbp_yg` and `dbp15k` directories contain datasets for entity alignment tasks. These datasets are commonly used in research for evaluating entity alignment models. The files within these directories follow a consistent naming convention that helps in identifying their purpose and the knowledge graph (KG) they belong to.

## Convention

The general naming convention for files is `{content_description}_{kg_identifier}`.

### Content Description

The prefix of the filename describes the data it contains. Common prefixes include:

-   `triples`: These files contain the relational triples (subject, predicate, object) that form the structure of the knowledge graph.
-   `attr` or `attr_triples`: These files store attribute information for the entities in the KG. This is often in the form of (entity, attribute_name, attribute_value).
-   `ent_links`: This file is crucial for the entity alignment task. It contains the ground truth mappings, i.e., pairs of equivalent entities between the two KGs.
-   `uri_attr_range_type` or `all_attrs_range`: These files provide schema-level information, such as the range or type of attributes.
-   `labels`: Files with `labels` in their name (e.g., `s_labels`, `t_labels`) usually contain human-readable labels or names for entities.

### KG Identifier

The suffix of the filename, typically `_1` or `_2`, identifies which of the two knowledge graphs the file belongs to.

-   `_1`: Refers to the first knowledge graph (source KG).
-   `_2`: Refers to the second knowledge graph (target KG).

Files that describe the relationship between the two KGs, such as `ent_links`, do not have a KG identifier suffix because they apply to both.

## Example from `dbp_yg`

-   `triples_1`: Relational triples for the first KG.
-   `triples_2`: Relational triples for the second KG.
-   `attr_triples_1`: Attribute triples for entities in the first KG.
-   `attr_triples_2`: Attribute triples for entities in the second KG.
-   `ent_links`: Pairs of equivalent entities between the two KGs.
