/**
 * WebLLM Toggle - Feature toggle for browser-based AI
 */

import React from 'react';
import { useWebLLM } from '../context/WebLLMContext';
import { Switch } from './ui/switch';
import { AlertCircle, CheckCircle, Loader2, Cpu } from 'lucide-react';

export const WebLLMToggle = () => {
  const { isEnabled, setIsEnabled, isLoading, isInitialized, error, loadingProgress } = useWebLLM();

  return (
    <div className="p-3 bg-si-surface/50 border border-si-border rounded-lg">
      <div className="flex items-center justify-between gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <Cpu size={14} className="text-si-laser flex-shrink-0" />
            <span className="text-xs font-mono uppercase tracking-wider text-si-muted truncate">
              Browser AI (Phi-3)
            </span>
            {isInitialized && (
              <CheckCircle className="w-3 h-3 text-green-500 flex-shrink-0" />
            )}
            {isLoading && (
              <Loader2 className="w-3 h-3 animate-spin text-si-laser flex-shrink-0" />
            )}
            {error && !isLoading && (
              <AlertCircle className="w-3 h-3 text-red-500 flex-shrink-0" />
            )}
          </div>
          <p className="text-[10px] text-si-muted/70 truncate">
            {isLoading && loadingProgress 
              ? loadingProgress.text
              : isInitialized 
                ? 'Running locally in browser' 
                : 'Enable for local AI inference'}
          </p>
          {isLoading && loadingProgress && (
            <div className="w-full bg-si-border rounded-full h-1 mt-2">
              <div
                className="bg-si-laser h-1 rounded-full transition-all"
                style={{ width: `${(loadingProgress.progress || 0) * 100}%` }}
              />
            </div>
          )}
        </div>
        
        <Switch
          checked={isEnabled}
          onCheckedChange={setIsEnabled}
          disabled={isLoading}
          className="flex-shrink-0"
        />
      </div>

      {error && (
        <div className="mt-2 p-2 bg-red-500/10 border border-red-500/30 rounded text-xs text-red-400">
          <AlertCircle className="w-3 h-3 inline mr-1" />
          {error}
        </div>
      )}
    </div>
  );
};

export default WebLLMToggle;
