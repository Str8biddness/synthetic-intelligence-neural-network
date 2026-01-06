/**
 * WebLLM Context - Provides browser-based LLM capabilities using Phi-3-mini
 * Runs entirely in the browser with no server-side processing
 */

import React, { createContext, useState, useCallback, useRef, useContext } from 'react';
import * as webllm from '@mlc-ai/web-llm';

const WEBLLM_CONFIG = {
  model: 'Phi-3-mini-4k-instruct-q4f32_1-MLC',
  chatOptions: {
    temperature: 0.7,
    top_p: 0.95,
    max_tokens: 512,
  },
  systemPrompt: `You are a helpful AI assistant running in the browser. You provide concise, accurate responses. Keep responses focused and practical.`
};

const WebLLMContext = createContext(undefined);

export const WebLLMProvider = ({ children }) => {
  const [engine, setEngine] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);
  const [error, setError] = useState(null);
  const [loadingProgress, setLoadingProgress] = useState(null);
  const [isEnabled, setIsEnabledState] = useState(() => {
    const saved = localStorage.getItem('webllm_enabled');
    return saved ? JSON.parse(saved) : false;
  });
  
  const engineRef = useRef(null);
  const initializingRef = useRef(false);

  const initializeEngine = useCallback(async (modelId = WEBLLM_CONFIG.model) => {
    if (initializingRef.current || isInitialized) {
      return;
    }

    initializingRef.current = true;
    setIsLoading(true);
    setError(null);
    setLoadingProgress({ text: 'Starting WebLLM engine...', progress: 0 });

    try {
      const initProgressCallback = (progress) => {
        setLoadingProgress({
          text: progress.text || 'Loading model...',
          progress: progress.progress || 0
        });
      };

      const newEngine = await webllm.CreateMLCEngine(
        modelId,
        { initProgressCallback }
      );

      engineRef.current = newEngine;
      setEngine(newEngine);
      setIsInitialized(true);
      setLoadingProgress(null);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to initialize WebLLM engine';
      setError(errorMessage);
      console.error('WebLLM initialization error:', err);
      setEngine(null);
      setIsInitialized(false);
    } finally {
      setIsLoading(false);
      initializingRef.current = false;
    }
  }, [isInitialized]);

  const releaseEngine = useCallback(async () => {
    if (engineRef.current) {
      engineRef.current = null;
      setEngine(null);
      setIsInitialized(false);
    }
  }, []);

  const setIsEnabled = useCallback((enabled) => {
    setIsEnabledState(enabled);
    localStorage.setItem('webllm_enabled', JSON.stringify(enabled));
    
    if (enabled && !isInitialized && !isLoading) {
      initializeEngine();
    }
  }, [isInitialized, isLoading, initializeEngine]);

  const generateResponse = useCallback(async (messages) => {
    if (!engineRef.current || !isInitialized) {
      throw new Error('Engine not initialized');
    }

    const formattedMessages = [
      { role: 'system', content: WEBLLM_CONFIG.systemPrompt },
      ...messages
    ];

    const response = await engineRef.current.chat.completions.create({
      messages: formattedMessages,
      temperature: WEBLLM_CONFIG.chatOptions.temperature,
      top_p: WEBLLM_CONFIG.chatOptions.top_p,
      max_tokens: WEBLLM_CONFIG.chatOptions.max_tokens,
      stream: false
    });

    return response.choices[0]?.message?.content || '';
  }, [isInitialized]);

  const generateStreamingResponse = useCallback(async function* (messages) {
    if (!engineRef.current || !isInitialized) {
      throw new Error('Engine not initialized');
    }

    const formattedMessages = [
      { role: 'system', content: WEBLLM_CONFIG.systemPrompt },
      ...messages
    ];

    const chunks = await engineRef.current.chat.completions.create({
      messages: formattedMessages,
      temperature: WEBLLM_CONFIG.chatOptions.temperature,
      top_p: WEBLLM_CONFIG.chatOptions.top_p,
      max_tokens: WEBLLM_CONFIG.chatOptions.max_tokens,
      stream: true
    });

    for await (const chunk of chunks) {
      const content = chunk.choices[0]?.delta?.content || '';
      if (content) {
        yield content;
      }
    }
  }, [isInitialized]);

  const value = {
    engine,
    isLoading,
    isInitialized,
    error,
    loadingProgress,
    initializeEngine,
    releaseEngine,
    isEnabled,
    setIsEnabled,
    generateResponse,
    generateStreamingResponse,
    config: WEBLLM_CONFIG
  };

  return (
    <WebLLMContext.Provider value={value}>
      {children}
    </WebLLMContext.Provider>
  );
};

export const useWebLLM = () => {
  const context = useContext(WebLLMContext);
  if (!context) {
    throw new Error('useWebLLM must be used within WebLLMProvider');
  }
  return context;
};

export default WebLLMContext;
