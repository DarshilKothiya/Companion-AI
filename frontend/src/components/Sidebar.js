import React, { useState, useEffect } from 'react';
import { MessageSquare, Plus, PanelLeft } from 'lucide-react';
import { chatAPI } from '../services/api';
import './Sidebar.css';

const Sidebar = ({ currentConversationId, onSelectConversation, onNewChat, isOpen, toggleSidebar }) => {
    const [conversations, setConversations] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [isCollapsed, setIsCollapsed] = useState(false);

    useEffect(() => {
        loadConversations();
    }, [currentConversationId]);

    const loadConversations = async () => {
        setIsLoading(true);
        try {
            const data = await chatAPI.listConversations(20, 0);
            if (Array.isArray(data)) {
                setConversations(data);
            } else if (data && data.items) {
                setConversations(data.items);
            }
        } catch (error) {
            console.error('Error loading conversations:', error);
        } finally {
            setIsLoading(false);
        }
    };

    const getConvLabel = (conv) => {
        if (conv.device_type) return `${conv.brand || ''} ${conv.device_type}`.trim();
        return new Date(conv.created_at || conv.updated_at).toLocaleDateString();
    };

    return (
        <>
            {/* Mobile backdrop */}
            <div className={`sidebar-overlay ${isOpen ? 'show' : ''}`} onClick={toggleSidebar} />

            <div className={`sidebar ${isOpen ? 'open' : ''} ${isCollapsed ? 'collapsed' : ''} glass-effect`}>

                {/* ── Header ── */}
                <div className={`sidebar-header ${isCollapsed ? 'collapsed-header' : ''}`}>
                    {isCollapsed ? (
                        /* Collapsed: toggle button, then new chat */
                        <>
                            <button className="new-chat-icon-btn" onClick={onNewChat} title="New Chat">
                                <Plus size={20} />
                            </button>
                            <button className="sidebar-toggle-btn" onClick={() => setIsCollapsed(c => !c)} title="Expand sidebar">
                                <PanelLeft size={20} />
                            </button>
                        </>
                    ) : (
                        /* Expanded: full new-chat button + toggle */
                        <>
                            <button className="new-chat-button" onClick={onNewChat}>
                                <Plus size={18} />
                                <span>New Chat</span>
                            </button>
                            <button className="sidebar-toggle-btn" onClick={() => setIsCollapsed(c => !c)} title="Collapse sidebar">
                                <PanelLeft size={20} />
                            </button>
                        </>
                    )}
                </div>

                {/* ── Body ── */}
                <div className="sidebar-content">
                    {!isCollapsed && (
                        <p className="sidebar-title">Recent Chats</p>
                    )}

                    {isLoading ? (
                        <div className="sidebar-loading">{isCollapsed ? '…' : 'Loading...'}</div>
                    ) : conversations.length === 0 ? (
                        <div className="sidebar-empty">
                            {!isCollapsed && 'No recent conversations'}
                        </div>
                    ) : (
                        <ul className="conversation-list">
                            {conversations.map((conv) => (
                                <li key={conv.conversation_id}>
                                    <button
                                        className={`conversation-item ${currentConversationId === conv.conversation_id ? 'active' : ''} ${isCollapsed ? 'icon-only' : ''}`}
                                        onClick={() => onSelectConversation(conv.conversation_id)}
                                        title={getConvLabel(conv)}
                                    >
                                        <MessageSquare size={18} className="conv-icon" />
                                        {!isCollapsed && (
                                            <span className="conversation-title">
                                                {getConvLabel(conv)}
                                            </span>
                                        )}
                                    </button>
                                </li>
                            ))}
                        </ul>
                    )}
                </div>
            </div>
        </>
    );
};

export default Sidebar;
