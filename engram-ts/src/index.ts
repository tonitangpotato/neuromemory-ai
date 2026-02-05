export { MemoryEntry, MemoryType, MemoryLayer, DEFAULT_DECAY_RATES, DEFAULT_IMPORTANCE } from './core';
export { MemoryConfig } from './config';
export { SQLiteStore } from './store';
export { Memory } from './memory';
export { baseLevelActivation, spreadingActivation, retrievalActivation, retrieveTopK } from './activation';
export { retrievability, computeStability, effectiveStrength, shouldForget, pruneforgotten, retrievalInducedForgetting } from './forgetting';
export { applyDecay, consolidateSingle, runConsolidationCycle, getConsolidationStats } from './consolidation';
export { contentReliability, retrievalSalience, confidenceScore, confidenceLabel, confidenceDetail } from './confidence';
export { detectFeedback, applyReward } from './reward';
export { synapticDownscale } from './downscaling';
export { BaselineTracker } from './anomaly';
export { SearchEngine, SearchResult } from './search';
export { recordCoactivation, maybeCreateLink, getHebbianNeighbors, decayHebbianLinks, strengthenLink, getAllHebbianLinks } from './hebbian';
export { SessionWorkingMemory, SessionRecallResult, getSessionWM, clearSession, listSessions } from './session_wm';

// Embedding exports (v1.0.0)
export { EmbeddingProvider, EmbeddingConfig, EmbeddingResult, ProviderInfo, DEFAULT_EMBEDDING_CONFIG } from './embeddings/base';
export { OpenAIEmbeddingProvider } from './embeddings/openai';
export { OllamaEmbeddingProvider } from './embeddings/ollama';
export { MCPEmbeddingProvider } from './embeddings/mcp';
export { detectProvider, getAvailableProviders } from './embeddings/provider_detection';
export { cosineSimilarity, vectorSearch, VectorSearchResult, migrateVectorColumn, storeVector, getVector, getVectorCount } from './vector_search';
export { hybridSearch, adaptiveHybridSearch, HybridSearchResult } from './hybrid_search';
