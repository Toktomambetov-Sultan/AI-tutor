import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import api from '../../api/axios';
import './Student.css';

export default function LessonViewer() {
    const { id: courseId, lessonId } = useParams();
    const navigate = useNavigate();
    const [lesson, setLesson] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => { fetchLesson(); }, [lessonId]);

    const fetchLesson = async () => {
        try {
            const response = await api.get(`/lessons/${lessonId}`);
            setLesson(response.data);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to fetch lesson');
        } finally {
            setLoading(false);
        }
    };

    const handleAiCall = () => {
        navigate(`/courses/${courseId}/lessons/${lessonId}/call`);
    };

    if (loading) return <div className="loading-state"><div className="spinner" /><span>Loading lesson...</span></div>;
    if (!lesson) return <div className="alert alert-error">{error || 'Lesson not found'}</div>;

    return (
        <div className="lesson-viewer lesson-viewer-layout">
            <div className="lesson-topbar">
                <button className="btn-back" onClick={() => navigate(`/courses/${courseId}`)}>
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M10 12L6 8l4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
                    Back to Course
                </button>
                <h1>{lesson.title}</h1>
            </div>

            <div className="lesson-columns">
                <section className="lesson-main-column">
                    {!lesson.materials?.length ? (
                        <div className="empty-state-card" style={{ marginTop: '1rem' }}>
                            <div className="empty-icon">📝</div>
                            <h3>No materials yet</h3>
                            <p>The teacher hasn't added content to this lesson yet.</p>
                        </div>
                    ) : (
                        lesson.materials.map((material, index) => (
                            <div key={material.id} className="card material-block">
                                <h3>
                                    {material.type === 'text' ? 'Text material' : 'PDF document'}
                                    {lesson.materials.length > 1 && ` #${index + 1}`}
                                </h3>
                                {material.type === 'text' ? (
                                    <div className="text-content">{material.content}</div>
                                ) : (
                                    <a
                                        href={`/api/v1/materials/${material.id}`}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="pdf-link"
                                    >
                                        Open PDF
                                    </a>
                                )}
                            </div>
                        ))
                    )}
                </section>

                <aside className="lesson-side-column">
                    <div className="ai-section">
                        <div className="ai-section-icon">🤖</div>
                        <h3>AI Tutor</h3>
                        <p>Start a live voice session that follows this lesson content.</p>
                        <button className="btn-primary" onClick={handleAiCall}>
                            Start Voice Session
                        </button>
                    </div>

                    <div className="card lesson-stats-card">
                        <h4>Lesson Snapshot</h4>
                        <p>{lesson.materials?.length || 0} materials ready</p>
                    </div>
                </aside>
            </div>
        </div>
    );
}
