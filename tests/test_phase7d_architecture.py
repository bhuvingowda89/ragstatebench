from ragstate.campaign.evidence_spans import claim_span, mapping_for_queries, numeric_pair_boundary
from ragstate.retrieval.architecture import (HierarchicalRetriever,
    chunks_for_architecture, section_aware_chunks)
from ragstate.retrieval.chunking import chunk_document
from ragstate.world.ledger import WorldLedger
from ragstate.world.models import DocumentVersion, Principal
from ragstate.world.scientific_v3 import V3Query


class ToyEncoder:
    def encode_documents(self, texts, batch_size):
        return [self._vec(text) for text in texts]

    def encode_query(self, texts, batch_size):
        return [self._vec(text) for text in texts]

    def encode(self, texts, batch_size):
        return self.encode_documents(texts, batch_size)

    @staticmethod
    def _vec(text):
        return [float(text.lower().count('alpha') + 1),
                float(text.lower().count('beta') + 1)]


def _world():
    world = WorldLedger()
    world.add_principal(Principal('p', 'tenant_a'))
    world.add_document(DocumentVersion('alpha-v1','alpha','tenant_a',1,0,
        content='Alpha operations.\n\nPurpose\n'+'alpha '*95+'\n\nRule\n'+'alpha '*110))
    world.add_document(DocumentVersion('beta-v1','beta','tenant_a',1,0,
        content='Beta operations.\n\nPurpose\n'+'beta '*95+'\n\nRule\n'+'beta '*110))
    return world


def test_160_word_chunks_preserve_provenance_without_overlap():
    world=_world()
    doc=world.document_at('alpha',0)
    chunks=chunk_document(doc,max_words=160)
    assert len(chunks)==2
    assert sum(len(c.text.split()) for c in chunks)==len(doc.content.split())
    assert all(c.document_id==doc.document_id and c.version==doc.version and
               c.tenant_id==doc.tenant_id and c.acl_version==0 for c in chunks)
    _,ra1=chunks_for_architecture(world,0,'RA1')
    assert len([c for c in ra1 if c.document_id=='alpha-v1'])==2


def test_section_aware_chunks_are_bounded_and_preserve_source_words():
    doc=_world().document_at('alpha',0)
    chunks=section_aware_chunks(doc,maximum_words=200)
    assert len(chunks)>=2
    assert max(len(c.text.split()) for c in chunks)<=200
    assert ' '.join(c.text for c in chunks)==' '.join(doc.content.split())
    assert all(c.chunk_id.startswith('alpha-v1:s') and c.logical_document_id=='alpha'
               for c in chunks)
    assert [c.chunk_id for c in chunks]==[c.chunk_id for c in section_aware_chunks(doc,200)]


def test_hierarchical_rank_restricts_chunks_to_selected_documents_without_gold():
    world=_world();docs,chunks=chunks_for_architecture(world,0,'RA3')
    fusion={'sparse_weight':1.0,'dense_weight':1.0,'rrf_k':60,'candidate_k':100}
    retriever=HierarchicalRetriever(docs,chunks,ToyEncoder(),'toy','fixed',fusion,document_top_n=1)
    first=retriever.retrieve('alpha',method='bm25',top_k=10)
    assert retriever.last_document_ids==('alpha-v1',)
    assert first and {h.chunk.document_id for h in first}=={'alpha-v1'}
    second=retriever.retrieve('alpha',method='bm25',top_k=10)
    assert [h.chunk.chunk_id for h in first]==[h.chunk.chunk_id for h in second]
    for method in ('e5','hybrid'):
        one=retriever.retrieve('alpha',method,10)
        two=retriever.retrieve('alpha',method,10)
        assert [h.chunk.chunk_id for h in one]==[h.chunk.chunk_id for h in two]


def test_exact_claim_span_mapping_exposes_boundary_crossing():
    prefix='background '*77
    sentence='For domestic employees, the standard rule governing travel meals sets the reimbursement cap at 42 dollars.'
    doc=DocumentVersion('d-v1','d','tenant_a',1,0,content=prefix+sentence)
    meta={'axis_one':'domestic','axis_two':'employees'}
    claim={'document_id':'d-v1','value':42,'unit':'dollars','claim_type':'default_rule'}
    span=claim_span(doc,meta,claim)
    assert span['span_text']==sentence
    world=WorldLedger();world.add_principal(Principal('p','tenant_a'));world.add_document(doc)
    query=V3Query('q','What is the cap?','hr','natural_descriptive','p','tenant_a',0,
                  ('d',),('d-v1',),(claim,),True,False,False,False,'UPDATE','d-v1',0,None,'single_source')
    chunk80=chunk_document(doc,max_words=80)
    chunk160=chunk_document(doc,max_words=160)
    section_chunks=section_aware_chunks(doc,maximum_words=200)
    assert not mapping_for_queries(world,{'d':meta},[query],chunk80)[0]['complete_span_available']
    assert mapping_for_queries(world,{'d':meta},[query],chunk160)[0]['complete_span_available']
    assert mapping_for_queries(world,{'d':meta},[query],section_chunks)[0]['complete_span_available']
    assert numeric_pair_boundary(doc,chunk80,42,'dollars') is False
