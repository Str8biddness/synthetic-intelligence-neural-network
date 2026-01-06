import { useState, useRef, useEffect } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import axios from "axios";
import { Toaster } from "sonner";
import { toast } from "sonner";
import { 
  Brain, 
  Send, 
  Zap, 
  Activity, 
  Database, 
  GitBranch,
  Sparkles,
  ChevronRight,
  RotateCcw,
  Info,
  Image,
  Loader2,
  Download,
  Pencil
} from "lucide-react";
import { ScrollArea } from "./components/ui/scroll-area";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "./components/ui/tooltip";
import { WebLLMProvider, useWebLLM } from "./context/WebLLMContext";
import WebLLMToggle from "./components/WebLLMToggle";
import "@/App.css";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Decoding text effect component
const DecodingText = ({ text, speed = 20 }) => {
  const [displayText, setDisplayText] = useState("");
  const [isComplete, setIsComplete] = useState(false);
  
  useEffect(() => {
    if (!text) return;
    
    let index = 0;
    setDisplayText("");
    setIsComplete(false);
    
    const interval = setInterval(() => {
      if (index < text.length) {
        setDisplayText(text.slice(0, index + 1));
        index++;
      } else {
        setIsComplete(true);
        clearInterval(interval);
      }
    }, speed);
    
    return () => clearInterval(interval);
  }, [text, speed]);
  
  return (
    <span className={isComplete ? "" : "border-r-2 border-si-laser"}>
      {displayText}
    </span>
  );
};

// Confidence meter component
const ConfidenceMeter = ({ confidence }) => {
  const percentage = Math.round(confidence * 100);
  
  return (
    <div className="flex items-center gap-3">
      <div className="confidence-bar w-24 rounded-sm">
        <div 
          className="confidence-fill" 
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className="font-mono text-sm text-si-laser">{percentage}%</span>
    </div>
  );
};

// Stats panel component
const StatsPanel = ({ stats, onRefresh }) => {
  if (!stats) return null;
  
  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between p-4 border-b border-si-border">
        <h2 className="font-heading text-sm uppercase tracking-wider text-si-muted">
          SI Statistics
        </h2>
        <button 
          onClick={onRefresh}
          className="text-si-muted hover:text-si-laser transition-colors"
          data-testid="refresh-stats-btn"
        >
          <RotateCcw size={14} />
        </button>
      </div>
      
      <ScrollArea className="flex-1 p-4">
        <div className="space-y-4">
          {/* Pattern Stats */}
          <div className="stat-card">
            <div className="flex items-center gap-2 mb-3">
              <Database size={14} className="text-si-laser" />
              <span className="text-xs uppercase tracking-wider text-si-muted">Patterns</span>
            </div>
            <div className="stat-value">{stats.patterns?.total || 0}</div>
            <div className="stat-label">Total Patterns</div>
            
            {stats.patterns?.domains && (
              <div className="mt-3 flex flex-wrap gap-1">
                {Object.entries(stats.patterns.domains).map(([domain, count]) => (
                  <span key={domain} className="domain-tag">
                    {domain}: {count}
                  </span>
                ))}
              </div>
            )}
          </div>
          
          {/* Entity Stats */}
          <div className="stat-card">
            <div className="flex items-center gap-2 mb-3">
              <Sparkles size={14} className="text-si-indigo" />
              <span className="text-xs uppercase tracking-wider text-si-muted">Entities</span>
            </div>
            <div className="stat-value text-si-indigo">{stats.entities?.total || 0}</div>
            <div className="stat-label">Knowledge Entities</div>
          </div>
          
          {/* Performance */}
          <div className="stat-card">
            <div className="flex items-center gap-2 mb-3">
              <Activity size={14} className="text-si-alert" />
              <span className="text-xs uppercase tracking-wider text-si-muted">Performance</span>
            </div>
            <div className="space-y-2 text-sm font-mono">
              <div className="flex justify-between">
                <span className="text-si-muted">Queries:</span>
                <span>{stats.performance?.total_queries || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-si-muted">Avg Time:</span>
                <span>{(stats.performance?.avg_response_time_ms || 0).toFixed(0)}ms</span>
              </div>
              <div className="flex justify-between">
                <span className="text-si-muted">Confidence:</span>
                <span>{((stats.performance?.avg_confidence || 0) * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
          
          {/* Consciousness */}
          {stats.consciousness && (
            <div className="stat-card tracing-border">
              <div className="flex items-center gap-2 mb-3">
                <Brain size={14} className="text-si-laser animate-pulse" />
                <span className="text-xs uppercase tracking-wider text-si-muted">Consciousness</span>
              </div>
              <div className="space-y-2 text-xs font-mono">
                <div className="flex justify-between">
                  <span className="text-si-muted">Mode:</span>
                  <span className="text-si-laser">{stats.consciousness.reasoning_mode}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-si-muted">Load:</span>
                  <span>{(stats.consciousness.cognitive_load * 100).toFixed(0)}%</span>
                </div>
              </div>
            </div>
          )}
          
          {/* WebLLM Toggle */}
          <WebLLMToggle />
        </div>
      </ScrollArea>
    </div>
  );
};

// Reasoning panel component
const ReasoningPanel = ({ currentResponse }) => {
  if (!currentResponse) {
    return (
      <div className="h-full flex flex-col">
        <div className="flex items-center justify-between p-4 border-b border-si-border">
          <h2 className="font-heading text-sm uppercase tracking-wider text-si-muted">
            Reasoning
          </h2>
        </div>
        <div className="flex-1 flex items-center justify-center text-si-muted text-sm">
          <span>Ask a question to see reasoning</span>
        </div>
      </div>
    );
  }
  
  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between p-4 border-b border-si-border">
        <h2 className="font-heading text-sm uppercase tracking-wider text-si-muted">
          Reasoning Analysis
        </h2>
      </div>
      
      <ScrollArea className="flex-1 p-4">
        <div className="space-y-4">
          {/* Strategy */}
          <div>
            <span className="text-xs text-si-muted uppercase tracking-wider">Strategy</span>
            <div className="mt-1">
              <span className="strategy-badge">
                <GitBranch size={12} />
                {currentResponse.reasoning_strategy}
              </span>
            </div>
          </div>
          
          {/* Confidence */}
          <div>
            <span className="text-xs text-si-muted uppercase tracking-wider">Confidence</span>
            <div className="mt-2">
              <ConfidenceMeter confidence={currentResponse.confidence} />
            </div>
          </div>
          
          {/* Domains */}
          {currentResponse.domains_involved?.length > 0 && (
            <div>
              <span className="text-xs text-si-muted uppercase tracking-wider">Domains</span>
              <div className="mt-1 flex flex-wrap gap-1">
                {currentResponse.domains_involved.map((domain, i) => (
                  <span key={i} className="domain-tag">{domain}</span>
                ))}
              </div>
            </div>
          )}
          
          {/* Patterns Used */}
          <div>
            <span className="text-xs text-si-muted uppercase tracking-wider">Patterns Matched</span>
            <div className="mt-1 font-mono text-lg text-si-laser">
              {currentResponse.patterns_used}
            </div>
          </div>
          
          {/* Response Time */}
          <div>
            <span className="text-xs text-si-muted uppercase tracking-wider">Response Time</span>
            <div className="mt-1 font-mono text-lg">
              {currentResponse.response_time_ms.toFixed(0)}
              <span className="text-sm text-si-muted">ms</span>
            </div>
          </div>
          
          {/* Reasoning Steps */}
          {currentResponse.reasoning_steps?.length > 0 && (
            <div>
              <span className="text-xs text-si-muted uppercase tracking-wider">Reasoning Steps</span>
              <div className="mt-2 space-y-2">
                {currentResponse.reasoning_steps.map((step, i) => (
                  <div key={i} className="p-2 bg-si-surface border border-si-border text-xs">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-si-laser">#{step.step_number}</span>
                      <span className="text-si-muted">{step.description}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Insights */}
          {currentResponse.insights?.length > 0 && (
            <div>
              <span className="text-xs text-si-muted uppercase tracking-wider flex items-center gap-1">
                <Sparkles size={12} className="text-si-indigo" />
                Cross-Domain Insights
              </span>
              <div className="mt-2 space-y-2">
                {currentResponse.insights.map((insight, i) => (
                  <div key={i} className="p-2 bg-si-indigo/10 border border-si-indigo/30 text-xs">
                    <div className="text-si-indigo">{insight.connection}</div>
                    <div className="mt-1 text-si-muted">{insight.explanation}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
};

// Visual description detection - checks if text describes something visual
const isVisualDescription = (text) => {
  if (!text) return false;
  
  const visualKeywords = [
    // Direct visual requests
    'picture', 'image', 'draw', 'paint', 'sketch', 'visualize', 'illustration',
    'show me', 'what does', 'look like', 'looks like', 'appearance', 'visual',
    'imagine', 'create', 'generate', 'render', 'depict', 'describe',
    // Scene types
    'scene', 'landscape', 'portrait', 'artwork', 'design', 'graphic', 'view',
    // Nature
    'sunset', 'sunrise', 'mountain', 'ocean', 'forest', 'city', 'building',
    'beach', 'lake', 'river', 'waterfall', 'cave', 'desert', 'valley', 'hill',
    // Objects
    'person', 'animal', 'flower', 'tree', 'sky', 'clouds', 'stars', 'moon', 'sun',
    'house', 'car', 'boat', 'plane', 'bird', 'fish', 'cat', 'dog',
    // Abstract/artistic
    'abstract', 'geometric', 'pattern', 'color', 'shape', 'circle', 'square',
    'triangle', 'spiral', 'wave', 'gradient', 'texture',
    // Weather/atmosphere  
    'rain', 'snow', 'fog', 'storm', 'lightning', 'rainbow', 'aurora',
    // Light/color
    'bright', 'dark', 'glow', 'shine', 'shadow', 'light', 'colorful'
  ];
  
  const lowerText = text.toLowerCase();
  return visualKeywords.some(keyword => lowerText.includes(keyword));
};

// Inline generated image component with action buttons
const InlineGeneratedImage = ({ imageData, prompt, onRegenerate, onEditPrompt }) => {
  const [isLoading, setIsLoading] = useState(false);
  
  const handleDownload = () => {
    if (!imageData?.png_base64) return;
    
    const link = document.createElement('a');
    link.href = `data:image/png;base64,${imageData.png_base64}`;
    link.download = `si-generated-${Date.now()}.png`;
    link.click();
    toast.success('Image downloaded!');
  };
  
  if (isLoading) {
    return (
      <div className="mt-3 p-4 border border-si-border rounded-sm bg-si-surface/30">
        <div className="flex items-center gap-2 text-si-laser">
          <Loader2 size={16} className="animate-spin" />
          <span className="text-xs font-mono uppercase">Generating image...</span>
        </div>
        <div className="mt-3 h-48 bg-si-surface/50 rounded animate-pulse flex items-center justify-center">
          <Sparkles size={24} className="text-si-muted" />
        </div>
      </div>
    );
  }
  
  if (!imageData) return null;
  
  return (
    <div className="mt-3 border border-si-border rounded-sm overflow-hidden bg-si-surface/30">
      {/* Image Header */}
      <div className="px-3 py-2 border-b border-si-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles size={12} className="text-si-laser" />
          <span className="text-xs font-mono uppercase text-si-muted">Generated Image</span>
        </div>
        <span className="text-[10px] text-si-muted font-mono">
          {imageData.timing?.total_ms?.toFixed(0) || '?'}ms
        </span>
      </div>
      
      {/* Image Display */}
      <div className="p-3">
        {imageData.png_base64 ? (
          <img 
            src={`data:image/png;base64,${imageData.png_base64}`}
            alt="Generated visualization"
            className="w-full max-w-md mx-auto rounded border border-si-border"
          />
        ) : imageData.svg ? (
          <div 
            className="w-full max-w-md mx-auto rounded border border-si-border bg-white p-2"
            dangerouslySetInnerHTML={{ __html: imageData.svg }}
          />
        ) : null}
      </div>
      
      {/* Action Buttons */}
      <div className="px-3 py-2 border-t border-si-border flex items-center gap-2">
        <button
          onClick={onRegenerate}
          className="flex items-center gap-1 px-2 py-1 text-xs font-mono uppercase text-si-muted hover:text-si-laser transition-colors border border-si-border rounded hover:border-si-laser"
        >
          <RotateCcw size={12} />
          Regenerate
        </button>
        <button
          onClick={onEditPrompt}
          className="flex items-center gap-1 px-2 py-1 text-xs font-mono uppercase text-si-muted hover:text-si-laser transition-colors border border-si-border rounded hover:border-si-laser"
        >
          <Pencil size={12} />
          Edit Prompt
        </button>
        <button
          onClick={handleDownload}
          className="flex items-center gap-1 px-2 py-1 text-xs font-mono uppercase text-si-muted hover:text-si-laser transition-colors border border-si-border rounded hover:border-si-laser"
        >
          <Download size={12} />
          Download
        </button>
      </div>
    </div>
  );
};

// Chat message component with inline image generation
const ChatMessage = ({ message, isUser, onGenerateImage, onEditPrompt }) => {
  const [generatedImage, setGeneratedImage] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [showGenerateButton, setShowGenerateButton] = useState(false);
  
  // Check if this AI response could have an image generated
  useEffect(() => {
    if (!isUser && message.content) {
      setShowGenerateButton(isVisualDescription(message.content));
    }
  }, [isUser, message.content]);
  
  const handleGenerateImage = async () => {
    if (isGenerating) return;
    setIsGenerating(true);
    
    try {
      const response = await axios.post(`${API}/generate-image`, {
        description: message.content,
        use_optimizer: true
      });
      
      setGeneratedImage(response.data);
      toast.success('Image generated!');
    } catch (error) {
      console.error('Image generation error:', error);
      toast.error('Failed to generate image');
    } finally {
      setIsGenerating(false);
    }
  };
  
  const handleRegenerate = async () => {
    setGeneratedImage(null);
    await handleGenerateImage();
  };
  
  const handleEditPrompt = () => {
    if (onEditPrompt) {
      onEditPrompt(message.content);
    }
  };
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div 
        className={`max-w-[85%] p-4 ${
          isUser 
            ? 'chat-user rounded-sm' 
            : 'chat-ai rounded-sm'
        }`}
        data-testid={isUser ? "user-message" : "ai-message"}
      >
        {!isUser && (
          <div className="flex items-center justify-between gap-2 mb-2">
            <div className="flex items-center gap-2">
              <Brain size={14} className="text-si-laser" />
              <span className="text-xs text-si-laser font-mono uppercase">SI Response</span>
            </div>
            
            {/* Generate Image Button */}
            {showGenerateButton && !generatedImage && !isGenerating && (
              <button
                onClick={handleGenerateImage}
                className="flex items-center gap-1 px-2 py-1 text-xs font-mono text-si-muted hover:text-si-laser transition-colors border border-si-border rounded hover:border-si-laser hover:bg-si-surface/50"
                title="Generate an image from this response"
              >
                <Sparkles size={12} />
                🎨 Generate Image
              </button>
            )}
            
            {isGenerating && (
              <div className="flex items-center gap-1 px-2 py-1 text-xs font-mono text-si-laser">
                <Loader2 size={12} className="animate-spin" />
                Generating...
              </div>
            )}
          </div>
        )}
        <div className={`${isUser ? 'font-mono text-sm' : 'text-sm leading-relaxed'}`}>
          {isUser ? message.content : <DecodingText text={message.content} speed={5} />}
        </div>
        
        {/* Inline Generated Image */}
        {!isUser && generatedImage && (
          <InlineGeneratedImage 
            imageData={generatedImage}
            prompt={message.content}
            onRegenerate={handleRegenerate}
            onEditPrompt={handleEditPrompt}
          />
        )}
      </div>
    </div>
  );
};

// Loading indicator
const LoadingIndicator = () => (
  <div className="flex justify-start mb-4">
    <div className="chat-ai rounded-sm p-4">
      <div className="flex items-center gap-2">
        <Brain size={14} className="text-si-laser animate-pulse" />
        <span className="text-xs text-si-laser font-mono uppercase">Processing</span>
      </div>
      <div className="flex gap-1 mt-2">
        <span className="loading-dot w-2 h-2 bg-si-laser rounded-full"></span>
        <span className="loading-dot w-2 h-2 bg-si-laser rounded-full"></span>
        <span className="loading-dot w-2 h-2 bg-si-laser rounded-full"></span>
      </div>
    </div>
  </div>
);

// Main Chat Interface
const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [currentResponse, setCurrentResponse] = useState(null);
  const [stats, setStats] = useState(null);
  const messagesEndRef = useRef(null);
  
  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  // Fetch stats
  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API}/si/stats`);
      setStats(response.data);
    } catch (error) {
      console.error("Error fetching stats:", error);
    }
  };
  
  useEffect(() => {
    fetchStats();
  }, []);
  
  // Send message
  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;
    
    const userMessage = { 
      id: Date.now().toString(),
      content: input, 
      isUser: true 
    };
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    
    try {
      const response = await axios.post(`${API}/si/ask`, {
        query: input
      });
      
      const aiMessage = {
        id: response.data.id,
        content: response.data.response,
        isUser: false
      };
      
      setMessages(prev => [...prev, aiMessage]);
      setCurrentResponse(response.data);
      fetchStats();
      
    } catch (error) {
      console.error("Error:", error);
      toast.error("Failed to process query");
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        content: "Error processing your query. Please try again.",
        isUser: false
      }]);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Handle key press
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };
  
  // Clear chat
  const clearChat = () => {
    setMessages([]);
    setCurrentResponse(null);
  };
  
  // Handle edit prompt from image generation
  const handleEditPrompt = (content) => {
    // Extract a shorter prompt for image generation from the content
    const shortPrompt = content.length > 100 
      ? content.substring(0, 100) + '...' 
      : content;
    setInput(`Generate image: ${shortPrompt}`);
    toast.info('Prompt loaded - modify and send to generate a new image');
  };
  
  return (
    <TooltipProvider>
      <div className="h-screen flex flex-col bg-si-black">
        {/* Header */}
        <header className="flex items-center justify-between px-6 py-4 border-b border-si-border">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 flex items-center justify-center border border-si-laser glow-laser">
              <Brain size={20} className="text-si-laser" />
            </div>
            <div>
              <h1 className="font-heading text-xl tracking-tight">
                Synthetic Intelligence
              </h1>
              <p className="text-xs text-si-muted font-mono">
                PATTERN-BASED REASONING ENGINE v1.0
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <Tooltip>
              <TooltipTrigger asChild>
                <a 
                  href="/generate"
                  className="text-si-muted hover:text-si-indigo transition-colors"
                  data-testid="image-gen-link"
                >
                  <Image size={18} />
                </a>
              </TooltipTrigger>
              <TooltipContent>Image Generation</TooltipContent>
            </Tooltip>
            
            <Tooltip>
              <TooltipTrigger asChild>
                <button 
                  onClick={clearChat}
                  className="text-si-muted hover:text-si-text transition-colors"
                  data-testid="clear-chat-btn"
                >
                  <RotateCcw size={18} />
                </button>
              </TooltipTrigger>
              <TooltipContent>Clear Chat</TooltipContent>
            </Tooltip>
            
            <div className="flex items-center gap-2 px-3 py-1 bg-si-laser-dim border border-si-laser/30">
              <Zap size={12} className="text-si-laser" />
              <span className="text-xs font-mono text-si-laser">ONLINE</span>
            </div>
          </div>
        </header>
        
        {/* Main content */}
        <div className="flex-1 flex overflow-hidden">
          {/* Stats Panel */}
          <aside className="w-64 border-r border-si-border si-panel hidden lg:flex flex-col">
            <StatsPanel stats={stats} onRefresh={fetchStats} />
          </aside>
          
          {/* Chat Area */}
          <main className="flex-1 flex flex-col min-w-0">
            {/* Messages */}
            <ScrollArea className="flex-1 p-6 grid-bg">
              {messages.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-center">
                  <div className="w-20 h-20 flex items-center justify-center border border-si-border mb-6">
                    <Brain size={40} className="text-si-muted" />
                  </div>
                  <h2 className="font-heading text-2xl mb-2">
                    Synthetic Intelligence Ready
                  </h2>
                  <p className="text-si-muted max-w-md">
                    Ask questions about science, philosophy, technology, or any domain.
                    The SI engine uses pure pattern-based reasoning without neural networks.
                  </p>
                  <div className="mt-8 flex flex-wrap gap-2 justify-center">
                    {["What is consciousness?", "Explain quantum mechanics", "How does gravity work?"].map((q) => (
                      <button
                        key={q}
                        onClick={() => setInput(q)}
                        className="px-3 py-2 text-xs font-mono border border-si-border hover:border-si-laser hover:text-si-laser transition-colors"
                        data-testid="suggestion-btn"
                      >
                        <ChevronRight size={12} className="inline mr-1" />
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="max-w-3xl mx-auto">
                  {messages.map((msg) => (
                    <ChatMessage 
                      key={msg.id} 
                      message={msg} 
                      isUser={msg.isUser}
                      onEditPrompt={handleEditPrompt}
                    />
                  ))}
                  {isLoading && <LoadingIndicator />}
                  <div ref={messagesEndRef} />
                </div>
              )}
            </ScrollArea>
            
            {/* Input Area */}
            <div className="p-4 border-t border-si-border">
              <div className="max-w-3xl mx-auto flex gap-3">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask the Synthetic Intelligence..."
                  className="flex-1 si-input px-4 py-3 text-sm"
                  disabled={isLoading}
                  data-testid="chat-input"
                />
                <button
                  onClick={sendMessage}
                  disabled={isLoading || !input.trim()}
                  className="si-btn flex items-center gap-2"
                  data-testid="send-btn"
                >
                  <Send size={16} />
                  <span className="hidden sm:inline">Send</span>
                </button>
              </div>
            </div>
          </main>
          
          {/* Reasoning Panel */}
          <aside className="w-80 border-l border-si-border si-panel hidden xl:flex flex-col">
            <ReasoningPanel currentResponse={currentResponse} />
          </aside>
        </div>
        
        <Toaster 
          position="bottom-right" 
          toastOptions={{
            style: {
              background: 'var(--si-panel)',
              border: '1px solid var(--si-border)',
              color: 'var(--si-text)',
            },
          }}
        />
      </div>
    </TooltipProvider>
  );
};

// Image Generation Component
const ImageGenerator = () => {
  const [prompt, setPrompt] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedSvg, setGeneratedSvg] = useState(null);
  const [generationStats, setGenerationStats] = useState(null);
  
  const generateImage = async () => {
    if (!prompt.trim() || isGenerating) return;
    
    setIsGenerating(true);
    setGeneratedSvg(null);
    
    try {
      const response = await axios.post(`${API}/generate-image`, {
        description: prompt
      });
      
      if (response.data.success) {
        setGeneratedSvg(response.data.svg);
        setGenerationStats({
          totalTime: response.data.total_time_ms,
          stageTimes: response.data.stage_times,
          cacheHit: response.data.cache_hit
        });
        toast.success(`Image generated in ${response.data.total_time_ms?.toFixed(0) || 0}ms`);
      } else {
        toast.error(response.data.error || "Failed to generate image");
      }
    } catch (error) {
      console.error("Error:", error);
      toast.error("Failed to generate image");
    } finally {
      setIsGenerating(false);
    }
  };
  
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      generateImage();
    }
  };
  
  return (
    <TooltipProvider>
      <div className="h-screen flex flex-col bg-si-black">
        {/* Header */}
        <header className="flex items-center justify-between px-6 py-4 border-b border-si-border">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 flex items-center justify-center border border-si-indigo" style={{boxShadow: '0 0 20px rgba(99, 102, 241, 0.3)'}}>
              <Image size={20} className="text-si-indigo" />
            </div>
            <div>
              <h1 className="font-heading text-xl tracking-tight">
                Pattern-Based Image Generation
              </h1>
              <p className="text-xs text-si-muted font-mono">
                NO NEURAL NETWORKS • PURE PATTERN COMPOSITION
              </p>
            </div>
          </div>
          
          <a href="/" className="si-btn text-xs">
            <Brain size={14} className="inline mr-2" />
            Chat Interface
          </a>
        </header>
        
        {/* Main content */}
        <div className="flex-1 flex overflow-hidden">
          {/* Input & Controls */}
          <aside className="w-80 border-r border-si-border si-panel flex flex-col">
            <div className="p-4 border-b border-si-border">
              <h2 className="font-heading text-sm uppercase tracking-wider text-si-muted mb-4">
                Describe Your Image
              </h2>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="e.g., a red car on a mountain road with sunset sky..."
                className="w-full h-32 bg-transparent border border-si-border p-3 text-sm resize-none focus:border-si-indigo focus:outline-none"
                disabled={isGenerating}
                data-testid="image-prompt-input"
              />
              <button
                onClick={generateImage}
                disabled={isGenerating || !prompt.trim()}
                className="w-full mt-3 si-btn flex items-center justify-center gap-2"
                data-testid="generate-image-btn"
                style={{borderColor: '#6366F1', color: '#6366F1'}}
              >
                {isGenerating ? (
                  <>
                    <Loader2 size={16} className="animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Sparkles size={16} />
                    Generate Image
                  </>
                )}
              </button>
            </div>
            
            {/* Sample prompts */}
            <div className="p-4 border-b border-si-border">
              <h3 className="text-xs text-si-muted uppercase tracking-wider mb-3">Sample Prompts</h3>
              <div className="space-y-2">
                {[
                  "sunset over ocean with sailboat",
                  "person walking dog in park",
                  "house with tree on sunny day",
                  "rainy mountain road with car"
                ].map((sample) => (
                  <button
                    key={sample}
                    onClick={() => setPrompt(sample)}
                    className="w-full text-left text-xs p-2 border border-si-border hover:border-si-indigo hover:text-si-indigo transition-colors"
                    data-testid="sample-prompt-btn"
                  >
                    <ChevronRight size={12} className="inline mr-1" />
                    {sample}
                  </button>
                ))}
              </div>
            </div>
            
            {/* Generation Stats */}
            {generationStats && (
              <div className="p-4">
                <h3 className="text-xs text-si-muted uppercase tracking-wider mb-3">Generation Stats</h3>
                <div className="space-y-2 text-xs font-mono">
                  <div className="flex justify-between">
                    <span className="text-si-muted">Total Time:</span>
                    <span className="text-si-laser">{generationStats.totalTime?.toFixed(0) || 0}ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-si-muted">Cache Hit:</span>
                    <span>{generationStats.cacheHit ? 'Yes' : 'No'}</span>
                  </div>
                  {generationStats.stageTimes && Object.entries(generationStats.stageTimes).map(([stage, time]) => (
                    <div key={stage} className="flex justify-between text-[10px]">
                      <span className="text-si-muted">{stage}:</span>
                      <span>{time?.toFixed(0) || 0}ms</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </aside>
          
          {/* Canvas Area */}
          <main className="flex-1 flex items-center justify-center p-8 grid-bg">
            {generatedSvg ? (
              <div className="bg-white rounded-sm shadow-2xl p-4" data-testid="generated-image-container">
                <div 
                  dangerouslySetInnerHTML={{ __html: generatedSvg }}
                  className="w-full h-full"
                />
              </div>
            ) : (
              <div className="text-center">
                <div className="w-32 h-32 mx-auto flex items-center justify-center border border-si-border mb-6">
                  <Image size={48} className="text-si-muted" />
                </div>
                <h2 className="font-heading text-xl mb-2 text-si-muted">
                  Pattern-Based Image Generation
                </h2>
                <p className="text-sm text-si-muted max-w-md">
                  Describe a scene and the SI engine will compose it using pure pattern matching.
                  No neural networks, no diffusion models — just intelligent pattern composition.
                </p>
              </div>
            )}
          </main>
        </div>
        
        <Toaster 
          position="bottom-right" 
          toastOptions={{
            style: {
              background: 'var(--si-panel)',
              border: '1px solid var(--si-border)',
              color: 'var(--si-text)',
            },
          }}
        />
      </div>
    </TooltipProvider>
  );
};

function App() {
  return (
    <WebLLMProvider>
      <div className="App">
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<ChatInterface />} />
            <Route path="/generate" element={<ImageGenerator />} />
          </Routes>
        </BrowserRouter>
      </div>
    </WebLLMProvider>
  );
}

export default App;
