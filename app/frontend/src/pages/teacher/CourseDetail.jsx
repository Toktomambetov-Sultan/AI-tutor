import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import api from '../../api/axios';
import './Teacher.css';

export default function CourseDetail() {
    const { id } = useParams();
    const navigate = useNavigate();
    const { t } = useTranslation();
    const [course, setCourse] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [successMsg, setSuccessMsg] = useState('');

    // Class form
    const [showClassForm, setShowClassForm] = useState(false);
    const [classTitle, setClassTitle] = useState('');
    const [classOrder, setClassOrder] = useState(0);

    // Lesson form - which class is open for adding
    const [lessonFormClassId, setLessonFormClassId] = useState(null);
    const [lessonTitle, setLessonTitle] = useState('');
    const [lessonOrder, setLessonOrder] = useState(0);

    // Materials
    const [materialLessonId, setMaterialLessonId] = useState(null);
    const [materialText, setMaterialText] = useState('');
    const [pdfFile, setPdfFile] = useState(null);
    const [lessonMaterials, setLessonMaterials] = useState({});

    // Preview
    const [previewMaterial, setPreviewMaterial] = useState(null);
    const [previewLoading, setPreviewLoading] = useState(false);

    // Expanded classes
    const [expandedClasses, setExpandedClasses] = useState({});

    const flash = (msg) => {
        setSuccessMsg(msg);
        setTimeout(() => setSuccessMsg(''), 3000);
    };

    const fetchCourse = async () => {
        try {
            const response = await api.get(`/courses/${id}`);
            setCourse(response.data);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to fetch course');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchCourse();
    }, [id]);

    const toggleClass = (classId) => {
        setExpandedClasses(prev => ({ ...prev, [classId]: !prev[classId] }));
    };

    // ─── Class CRUD ───
    const handleCreateClass = async (e) => {
        e.preventDefault();
        try {
            await api.post(`/courses/${id}/classes`, { title: classTitle, order: classOrder });
            setClassTitle('');
            setClassOrder(0);
            setShowClassForm(false);
            flash('Class created successfully');
            fetchCourse();
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to create class');
        }
    };

    const handleDeleteClass = async (classId) => {
        if (!confirm('Delete this class and all its lessons?')) return;
        try {
            await api.delete(`/classes/${classId}`);
            flash('Class deleted');
            fetchCourse();
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to delete class');
        }
    };

    // ─── Lesson CRUD ───
    const handleCreateLesson = async (e, classId) => {
        e.preventDefault();
        try {
            await api.post(`/classes/${classId}/lessons`, { title: lessonTitle, order: lessonOrder });
            setLessonTitle('');
            setLessonOrder(0);
            setLessonFormClassId(null);
            flash('Lesson created successfully');
            setExpandedClasses(prev => ({ ...prev, [classId]: true }));
            fetchCourse();
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to create lesson');
        }
    };

    const handleDeleteLesson = async (lessonId) => {
        if (!confirm('Delete this lesson and its materials?')) return;
        try {
            await api.delete(`/lessons/${lessonId}`);
            flash('Lesson deleted');
            fetchCourse();
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to delete lesson');
        }
    };

    // ─── Materials ───
    const loadMaterials = async (lessonId) => {
        try {
            const response = await api.get(`/lessons/${lessonId}/materials`);
            setLessonMaterials(prev => ({ ...prev, [lessonId]: response.data.materials }));
        } catch (err) {
            console.error('Failed to load materials');
        }
    };

    const handleUploadText = async (e, lessonId) => {
        e.preventDefault();
        try {
            await api.post(`/lessons/${lessonId}/materials`, { content: materialText });
            setMaterialText('');
            flash('Text material added');
            loadMaterials(lessonId);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to add material');
        }
    };

    const handleUploadPdf = async (e, lessonId) => {
        e.preventDefault();
        if (!pdfFile) return;
        try {
            const formData = new FormData();
            formData.append('file', pdfFile);
            await api.post(`/lessons/${lessonId}/materials/upload`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            setPdfFile(null);
            flash('PDF uploaded');
            loadMaterials(lessonId);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to upload PDF');
        }
    };

    const handleDeleteMaterial = async (materialId, lessonId) => {
        try {
            await api.delete(`/materials/${materialId}`);
            loadMaterials(lessonId);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to delete material');
        }
    };

    const handlePreviewMaterial = async (mat) => {
        if (mat.type === 'text') {
            setPreviewMaterial({ ...mat, loading: false });
            return;
        }
        // PDF — open in new tab
        try {
            setPreviewLoading(true);
            const response = await api.get(`/teacher/materials/${mat.id}`, { responseType: 'blob' });
            const url = URL.createObjectURL(response.data);
            window.open(url, '_blank');
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to preview material');
        } finally {
            setPreviewLoading(false);
        }
    };

    const closePreview = () => setPreviewMaterial(null);

    if (loading) return <div className="loading-state"><div className="spinner" /><span>Loading course...</span></div>;
    if (!course) return <div className="alert alert-error">{error || 'Course not found'}</div>;

    const classes = course.classes || [];

    return (
        <div className="course-detail-page">
            <button className="btn-back" onClick={() => navigate('/teacher/courses')}>
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M10 12L6 8l4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
                Back to Courses
            </button>

            <div className="teacher-detail-layout">
                <aside className="teacher-detail-rail">
                    <div className="course-header-card">
                        <div className="course-header-info">
                            <h1>{course.title}</h1>
                            {course.description && <p className="course-description">{course.description}</p>}
                            <div className="course-stats">
                                <span className="stat-chip">📁 {classes.length} class{classes.length !== 1 ? 'es' : ''}</span>
                                <span className="stat-chip">📖 {classes.reduce((a, c) => a + (c.lessons?.length || 0), 0)} lessons</span>
                            </div>
                        </div>
                    </div>
                </aside>

                <section className="teacher-detail-main">
                    {error && <div className="alert alert-error">{error}<button className="alert-close" onClick={() => setError('')}>×</button></div>}
                    {successMsg && <div className="alert alert-success">{successMsg}</div>}

                    <div className="section">
                        <div className="section-header">
                            <h2>Classes &amp; Lessons</h2>
                            <button className="btn-primary btn-sm" onClick={() => setShowClassForm(!showClassForm)}>
                                {showClassForm ? 'Cancel' : '+ Add Class'}
                            </button>
                        </div>

                        {showClassForm && (
                            <div className="inline-form-card">
                                <form onSubmit={handleCreateClass}>
                                    <div className="inline-form-row">
                                        <div className="form-group" style={{ flex: 1 }}>
                                            <label>Class Title</label>
                                            <input type="text" value={classTitle} onChange={e => setClassTitle(e.target.value)} placeholder="e.g., Week 1: Introduction" required />
                                        </div>
                                        <div className="form-group" style={{ width: '100px' }}>
                                            <label>Order</label>
                                            <input type="number" value={classOrder} onChange={e => setClassOrder(parseInt(e.target.value) || 0)} />
                                        </div>
                                        <button type="submit" className="btn-primary btn-sm" style={{ alignSelf: 'flex-end', marginBottom: '1rem' }}>Create</button>
                                    </div>
                                </form>
                            </div>
                        )}

                        {classes.length === 0 ? (
                            <div className="empty-state-card">
                                <div className="empty-icon">📚</div>
                                <h3>No classes yet</h3>
                                <p>Create your first class to start building course content.</p>
                            </div>
                        ) : (
                            <div className="accordion-list">
                                {classes.map(cls => {
                                    const isExpanded = expandedClasses[cls.id];
                                    const lessons = cls.lessons || [];
                                    return (
                                        <div key={cls.id} className={`accordion-item ${isExpanded ? 'expanded' : ''}`}>
                                    <div className="accordion-header" onClick={() => toggleClass(cls.id)}>
                                        <div className="accordion-title-area">
                                            <svg className={`chevron ${isExpanded ? 'open' : ''}`} width="18" height="18" viewBox="0 0 16 16" fill="none"><path d="M6 4l4 4-4 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
                                            <span className="accordion-title">{cls.title}</span>
                                            <span className="accordion-badge">{lessons.length} lesson{lessons.length !== 1 ? 's' : ''}</span>
                                        </div>
                                        <div className="accordion-actions" onClick={e => e.stopPropagation()}>
                                            <button className="btn-ghost btn-sm" onClick={() => setLessonFormClassId(lessonFormClassId === cls.id ? null : cls.id)}>
                                                + Lesson
                                            </button>
                                            <button className="btn-ghost-danger btn-sm" onClick={() => handleDeleteClass(cls.id)} title="Delete class">
                                                ✕
                                            </button>
                                        </div>
                                    </div>

                                    {lessonFormClassId === cls.id && (
                                        <div className="accordion-inline-form">
                                            <form onSubmit={e => handleCreateLesson(e, cls.id)}>
                                                <div className="inline-form-row">
                                                    <div className="form-group" style={{ flex: 1 }}>
                                                        <input type="text" value={lessonTitle} onChange={e => setLessonTitle(e.target.value)} placeholder="Lesson title..." required />
                                                    </div>
                                                    <div className="form-group" style={{ width: '80px' }}>
                                                        <input type="number" value={lessonOrder} onChange={e => setLessonOrder(parseInt(e.target.value) || 0)} placeholder="#" />
                                                    </div>
                                                    <button type="submit" className="btn-primary btn-sm">Add</button>
                                                    <button type="button" className="btn-ghost btn-sm" onClick={() => setLessonFormClassId(null)}>Cancel</button>
                                                </div>
                                            </form>
                                        </div>
                                    )}

                                    {isExpanded && (
                                        <div className="accordion-body">
                                            {lessons.length === 0 ? (
                                                <p className="no-lessons-msg">No lessons yet. Click "+ Lesson" to add one.</p>
                                            ) : (
                                                lessons.map(lesson => (
                                                    <div key={lesson.id} className="lesson-row">
                                                        <div className="lesson-row-main">
                                                            <div className="lesson-row-info">
                                                                <span className="lesson-icon">📖</span>
                                                                <span className="lesson-name">{lesson.title}</span>
                                                                <span className="lesson-order">#{lesson.order}</span>
                                                            </div>
                                                            <div className="lesson-row-actions">
                                                                <button
                                                                    className={`btn-ghost btn-sm ${materialLessonId === lesson.id ? 'active' : ''}`}
                                                                    onClick={() => {
                                                                        const next = materialLessonId === lesson.id ? null : lesson.id;
                                                                        setMaterialLessonId(next);
                                                                        if (next && !lessonMaterials[lesson.id]) loadMaterials(lesson.id);
                                                                    }}
                                                                >
                                                                    📎 Materials
                                                                </button>
                                                                <button className="btn-ghost-danger btn-sm" onClick={() => handleDeleteLesson(lesson.id)}>✕</button>
                                                            </div>
                                                        </div>

                                                        {materialLessonId === lesson.id && (
                                                            <div className="materials-panel">
                                                                <h4>Materials</h4>
                                                                {lessonMaterials[lesson.id]?.length > 0 && (
                                                                    <div className="materials-list">
                                                                        {lessonMaterials[lesson.id].map(mat => (
                                                                            <div key={mat.id} className="material-chip">
                                                                                <span>{mat.type === 'text' ? '📝 Text content' : '📄 ' + (mat.file_path?.split('/').pop() || 'PDF')}</span>
                                                                                <div className="material-chip-actions">
                                                                                    <button className="btn-ghost btn-sm" onClick={() => handlePreviewMaterial(mat)} disabled={previewLoading} title="Preview">
                                                                                        👁️
                                                                                    </button>
                                                                                    <button className="btn-icon-danger" onClick={() => handleDeleteMaterial(mat.id, lesson.id)}>×</button>
                                                                                </div>
                                                                            </div>
                                                                        ))}
                                                                    </div>
                                                                )}
                                                                <div className="material-forms">
                                                                    <form onSubmit={e => handleUploadText(e, lesson.id)}>
                                                                        <textarea value={materialText} onChange={e => setMaterialText(e.target.value)} rows={3} placeholder="Enter text content..." />
                                                                        <button type="submit" className="btn-primary btn-sm" disabled={!materialText} style={{ marginTop: '0.5rem' }}>Add Text</button>
                                                                    </form>
                                                                    <form onSubmit={e => handleUploadPdf(e, lesson.id)} className="pdf-upload-form">
                                                                        <label className="file-input-label">
                                                                            <input type="file" accept=".pdf" onChange={e => setPdfFile(e.target.files[0])} />
                                                                            <span>{pdfFile ? pdfFile.name : 'Choose PDF file...'}</span>
                                                                        </label>
                                                                        <button type="submit" className="btn-primary btn-sm" disabled={!pdfFile}>Upload PDF</button>
                                                                    </form>
                                                                </div>
                                                            </div>
                                                        )}
                                                    </div>
                                                ))
                                            )}
                                        </div>
                                    )}
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </div>
                </section>
            </div>

            {/* Text Material Preview Modal */}
            {previewMaterial && previewMaterial.type === 'text' && (
                <div className="preview-overlay" onClick={closePreview}>
                    <div className="preview-modal" onClick={e => e.stopPropagation()}>
                        <div className="preview-modal-header">
                            <h3>📝 Text Material Preview</h3>
                            <button className="btn-ghost btn-sm" onClick={closePreview}>✕</button>
                        </div>
                        <div className="preview-modal-body">
                            <div className="preview-text-content">{previewMaterial.content}</div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
