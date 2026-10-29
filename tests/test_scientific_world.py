from ragstate.world.scientific import (
    BLUEPRINTS, FAMILIES, TRANSITIONS, generate_scientific_queries,
    generate_scientific_world,
)


def test_scientific_world_is_balanced_and_domain_specific():
    world = generate_scientific_world(101)
    queries = generate_scientific_queries(world, 202)
    assert len(world.documents) == 180
    assert len(queries) == 180
    assert len({q.text for q in queries}) == 180
    assert {q.domain for q in queries} == {d.name for d in BLUEPRINTS}
    for domain in BLUEPRINTS:
        subset = [q for q in queries if q.domain == domain.name]
        assert len(subset) == 30
        assert {q.family for q in subset} == set(FAMILIES)
        assert {q.transition_type for q in subset} == set(TRANSITIONS)
        assert all(q.logical_document_ids[0].startswith(domain.name.upper()) for q in subset)
    assert queries == generate_scientific_queries(world, 202)
    assert [q.principal_id for q in queries] != [q.principal_id for q in
            generate_scientific_queries(world, 203)]


def test_scientific_gold_is_not_derived_from_retrieval():
    world = generate_scientific_world(101)
    for query in generate_scientific_queries(world, 202):
        docs = [world.document_at(x, 9) for x in query.logical_document_ids]
        assert all(d is not None for d in docs)
        if query.family == "negative":
            assert not world.allowed(query.principal_id, query.logical_document_ids[0], 9)
        if query.family == "multi_document":
            assert len(docs) == 2
            assert docs[0].tenant_id == docs[1].tenant_id
