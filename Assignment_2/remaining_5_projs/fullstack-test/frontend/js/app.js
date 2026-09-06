// TodoPro Main Frontend Application
const API_BASE = window.location.origin;

class TodoApp {
    constructor() {
        this.tasks = [];
        this.tags = [];
        this.analytics = null;

        this.filterStatus = 'all'; // all, active, completed
        this.filterPriority = null; // null, P1, P2, P3, P4
        this.filterTagId = null; // null or tag id
        this.filterDue = null; // null, today, tomorrow, overdue, upcoming, no_date
        this.searchQuery = '';
        this.sortBy = 'position';
        this.sortOrder = 'asc';

        this.isDark = localStorage.getItem('todopro_dark') === 'true' || 
                      (!('todopro_dark' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches);

        this.draggedTaskId = null;
        this.editingTaskId = null;

        this.init();
    }

    async init() {
        this.applyTheme();
        this.bindGlobalShortcuts();
        this.bindUIEvents();
        await Promise.all([this.loadTags(), this.loadTasks(), this.loadAnalytics()]);
    }

    applyTheme() {
        if (this.isDark) {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
        localStorage.setItem('todopro_dark', this.isDark);
        const themeBtn = document.getElementById('theme-toggle-btn');
        if (themeBtn) {
            themeBtn.innerHTML = this.isDark ? '☀️ Light' : '🌙 Dark';
        }
    }

    toggleTheme() {
        this.isDark = !this.isDark;
        this.applyTheme();
        this.showToast(`Switched to ${this.isDark ? 'Dark' : 'Light'} Mode`);
    }

    // ================== API CALLS ==================

    async loadTasks() {
        try {
            const params = new URLSearchParams();
            if (this.filterStatus) params.append('status', this.filterStatus);
            if (this.filterPriority) params.append('priority', this.filterPriority);
            if (this.filterTagId) params.append('tag_id', this.filterTagId);
            if (this.filterDue) params.append('due_filter', this.filterDue);
            if (this.searchQuery) params.append('search', this.searchQuery);
            if (this.sortBy) params.append('sort_by', this.sortBy);
            if (this.sortOrder) params.append('order', this.sortOrder);

            const res = await fetch(`${API_BASE}/api/tasks?${params.toString()}`);
            if (!res.ok) throw new Error('Failed to fetch tasks');
            this.tasks = await res.json();
            this.renderTaskList();
            this.updateFilterCounts();
        } catch (err) {
            console.error(err);
            this.showToast('Error loading tasks', 'error');
        }
    }

    async loadTags() {
        try {
            const res = await fetch(`${API_BASE}/api/tags`);
            if (!res.ok) throw new Error('Failed to fetch tags');
            this.tags = await res.json();
            this.renderTagList();
            this.populateModalTags();
        } catch (err) {
            console.error(err);
        }
    }

    async loadAnalytics() {
        try {
            const res = await fetch(`${API_BASE}/api/analytics`);
            if (!res.ok) throw new Error('Failed to fetch analytics');
            this.analytics = await res.json();
            this.renderAnalyticsHeader();
            this.renderAnalyticsWidgets();
        } catch (err) {
            console.error(err);
        }
    }

    async createTask(taskData) {
        try {
            const res = await fetch(`${API_BASE}/api/tasks`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(taskData)
            });
            if (!res.ok) throw new Error('Failed to create task');
            const newTask = await res.json();
            this.showToast(`Task created: "${newTask.title}"`, 'success');
            await Promise.all([this.loadTasks(), this.loadAnalytics(), this.loadTags()]);
            return newTask;
        } catch (err) {
            console.error(err);
            this.showToast('Failed to create task', 'error');
        }
    }

    async updateTask(taskId, taskData) {
        try {
            const res = await fetch(`${API_BASE}/api/tasks/${taskId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(taskData)
            });
            if (!res.ok) throw new Error('Failed to update task');
            this.showToast('Task updated successfully', 'success');
            await Promise.all([this.loadTasks(), this.loadAnalytics(), this.loadTags()]);
        } catch (err) {
            console.error(err);
            this.showToast('Failed to update task', 'error');
        }
    }

    async toggleTask(taskId, clientX, clientY) {
        try {
            const res = await fetch(`${API_BASE}/api/tasks/${taskId}/toggle`, {
                method: 'POST'
            });
            if (!res.ok) throw new Error('Failed to toggle task');
            const updated = await res.json();

            if (updated.status === 'completed') {
                // Play harmonic success chime and trigger confetti
                if (window.soundFX) window.soundFX.playSuccessChime();
                if (window.triggerConfetti) window.triggerConfetti(clientX, clientY);
                this.showToast(`Completed! 🎉 "${updated.title}"`, 'success');
            } else {
                if (window.soundFX) window.soundFX.playSubtaskTick();
                this.showToast(`Marked active: "${updated.title}"`);
            }

            await Promise.all([this.loadTasks(), this.loadAnalytics()]);
        } catch (err) {
            console.error(err);
            this.showToast('Failed to toggle task status', 'error');
        }
    }

    async deleteTask(taskId, title) {
        if (!confirm(`Are you sure you want to delete "${title}"?`)) return;
        try {
            const res = await fetch(`${API_BASE}/api/tasks/${taskId}`, {
                method: 'DELETE'
            });
            if (!res.ok) throw new Error('Failed to delete task');
            if (window.soundFX) window.soundFX.playDeleteSwoosh();
            this.showToast(`Deleted task: "${title}"`);
            await Promise.all([this.loadTasks(), this.loadAnalytics(), this.loadTags()]);
        } catch (err) {
            console.error(err);
            this.showToast('Failed to delete task', 'error');
        }
    }

    async reorderTasks(orders) {
        try {
            await fetch(`${API_BASE}/api/tasks/reorder`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ orders })
            });
            this.showToast('Task order updated', 'info');
        } catch (err) {
            console.error(err);
        }
    }

    async addSubtask(taskId, title) {
        if (!title || !title.trim()) return;
        try {
            const res = await fetch(`${API_BASE}/api/tasks/${taskId}/subtasks`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: title.trim() })
            });
            if (!res.ok) throw new Error('Failed to add subtask');
            if (window.soundFX) window.soundFX.playSubtaskTick();
            await Promise.all([this.loadTasks(), this.loadAnalytics()]);
        } catch (err) {
            console.error(err);
            this.showToast('Failed to add subtask', 'error');
        }
    }

    async toggleSubtask(subtaskId, isCompleted) {
        try {
            const res = await fetch(`${API_BASE}/api/subtasks/${subtaskId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ is_completed: isCompleted })
            });
            if (!res.ok) throw new Error('Failed to update subtask');
            if (window.soundFX) window.soundFX.playSubtaskTick();
            await Promise.all([this.loadTasks(), this.loadAnalytics()]);
        } catch (err) {
            console.error(err);
        }
    }

    async deleteSubtask(subtaskId) {
        try {
            const res = await fetch(`${API_BASE}/api/subtasks/${subtaskId}`, {
                method: 'DELETE'
            });
            if (!res.ok) throw new Error('Failed to delete subtask');
            await Promise.all([this.loadTasks(), this.loadAnalytics()]);
        } catch (err) {
            console.error(err);
        }
    }

    async createTag(name, color) {
        if (!name || !name.trim()) return;
        try {
            const res = await fetch(`${API_BASE}/api/tags`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name.trim(), color: color || '#6366f1' })
            });
            if (!res.ok) throw new Error('Failed to create tag');
            this.showToast(`Tag created: #${name}`);
            await this.loadTags();
        } catch (err) {
            console.error(err);
            this.showToast('Failed to create tag', 'error');
        }
    }

    async deleteTag(tagId, name) {
        if (!confirm(`Delete tag #${name}?`)) return;
        try {
            const res = await fetch(`${API_BASE}/api/tags/${tagId}`, {
                method: 'DELETE'
            });
            if (!res.ok) throw new Error('Failed to delete tag');
            this.showToast(`Deleted tag #${name}`);
            if (this.filterTagId === tagId) this.filterTagId = null;
            await Promise.all([this.loadTags(), this.loadTasks()]);
        } catch (err) {
            console.error(err);
        }
    }

    // ================== RENDERING ==================

    renderAnalyticsHeader() {
        if (!this.analytics) return;
        const streakEl = document.getElementById('stat-streak');
        const scoreEl = document.getElementById('stat-score');
        const velocityEl = document.getElementById('stat-velocity');

        if (streakEl) streakEl.innerHTML = `🔥 ${this.analytics.current_streak}d streak`;
        if (scoreEl) scoreEl.innerHTML = `🎯 ${this.analytics.productivity_score} pts`;

        const todayVelocity = this.analytics.daily_velocity?.[this.analytics.daily_velocity.length - 1]?.completed_count || 0;
        if (velocityEl) velocityEl.innerHTML = `⚡ ${todayVelocity} today`;
    }

    renderAnalyticsWidgets() {
        if (!this.analytics || !window.AnalyticsRenderer) return;

        // Score gauge in sidebar
        AnalyticsRenderer.renderScoreGauge(this.analytics.productivity_score, 'sidebar-score-gauge');

        // Modal full charts
        AnalyticsRenderer.renderScoreGauge(this.analytics.productivity_score, 'modal-score-gauge');
        AnalyticsRenderer.renderVelocityChart(this.analytics.daily_velocity, 'velocity-chart-container');
        AnalyticsRenderer.renderTagBreakdown(this.analytics.tag_breakdown, 'tags-breakdown-container');

        // Modal stats
        const mTotal = document.getElementById('modal-total-tasks');
        const mComp = document.getElementById('modal-completed-tasks');
        const mRate = document.getElementById('modal-completion-rate');
        const mStreak = document.getElementById('modal-longest-streak');

        if (mTotal) mTotal.innerText = this.analytics.total_tasks;
        if (mComp) mComp.innerText = this.analytics.completed_tasks;
        if (mRate) mRate.innerText = `${this.analytics.completion_rate}%`;
        if (mStreak) mStreak.innerText = `${this.analytics.longest_streak} days`;
    }

    updateFilterCounts() {
        const total = this.tasks.length;
        const countHeader = document.getElementById('current-filter-count');
        if (countHeader) countHeader.innerText = `${total} task${total === 1 ? '' : 's'}`;
    }

    renderTagList() {
        const container = document.getElementById('tags-sidebar-list');
        if (!container) return;

        if (this.tags.length === 0) {
            container.innerHTML = `<span class="text-xs text-gray-400">No tags created</span>`;
            return;
        }

        container.innerHTML = this.tags.map(tag => {
            const isSelected = this.filterTagId === tag.id;
            return `
                <button onclick="window.app.setTagFilter(${tag.id})" 
                    class="w-full flex items-center justify-between px-2.5 py-1.5 rounded-lg text-xs font-medium transition-colors ${isSelected ? 'bg-indigo-50 text-indigo-700 dark:bg-indigo-950/40 dark:text-indigo-300 font-semibold' : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'}">
                    <span class="flex items-center space-x-2 truncate">
                        <span class="w-2.5 h-2.5 rounded-full flex-shrink-0" style="background-color: ${tag.color || '#6366f1'}"></span>
                        <span class="truncate">#${tag.name}</span>
                    </span>
                    <span class="text-[10px] text-gray-400 font-mono">${tag.task_count || 0}</span>
                </button>
            `;
        }).join('');
    }

    populateModalTags() {
        const container = document.getElementById('modal-tags-select-container');
        if (!container) return;

        container.innerHTML = this.tags.map(tag => `
            <label class="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium border cursor-pointer transition-colors border-gray-200 dark:border-gray-700 hover:border-indigo-400">
                <input type="checkbox" value="${tag.id}" class="task-modal-tag-checkbox mr-1.5 text-indigo-600 rounded">
                <span class="w-2 h-2 rounded-full mr-1.5" style="background-color: ${tag.color || '#6366f1'}"></span>
                <span>${tag.name}</span>
            </label>
        `).join('');
    }

    renderTaskList() {
        const container = document.getElementById('task-list-container');
        if (!container) return;

        if (this.tasks.length === 0) {
            container.innerHTML = `
                <div class="py-16 text-center">
                    <div class="w-16 h-16 bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-500 rounded-2xl flex items-center justify-center mx-auto mb-4 text-2xl">
                        ✓
                    </div>
                    <h3 class="text-base font-semibold text-gray-700 dark:text-gray-300">No tasks found</h3>
                    <p class="text-xs text-gray-400 mt-1 max-w-sm mx-auto">There are no tasks matching your current filter. Try selecting another filter or press <kbd class="px-1.5 py-0.5 bg-gray-200 dark:bg-gray-700 rounded font-mono">N</kbd> to add a new task.</p>
                </div>
            `;
            return;
        }

        container.innerHTML = this.tasks.map((task, idx) => this.renderTaskCard(task, idx)).join('');
        this.bindDragEvents();
    }

    renderTaskCard(task, idx) {
        const isCompleted = task.status === 'completed';

        // Priority styling
        const priorityConfig = {
            'P1': { label: 'P1 Urgent', bg: 'bg-red-50 dark:bg-red-950/30', text: 'text-red-700 dark:text-red-400', border: 'border-red-200 dark:border-red-800' },
            'P2': { label: 'P2 High', bg: 'bg-amber-50 dark:bg-amber-950/30', text: 'text-amber-700 dark:text-amber-400', border: 'border-amber-200 dark:border-amber-800' },
            'P3': { label: 'P3 Medium', bg: 'bg-indigo-50 dark:bg-indigo-950/30', text: 'text-indigo-700 dark:text-indigo-400', border: 'border-indigo-200 dark:border-indigo-800' },
            'P4': { label: 'P4 Low', bg: 'bg-gray-50 dark:bg-gray-800', text: 'text-gray-600 dark:text-gray-400', border: 'border-gray-200 dark:border-gray-700' }
        };
        const pConf = priorityConfig[task.priority] || priorityConfig['P3'];

        // Relative due badge styling
        let dueBadgeHtml = '';
        if (task.relative_due) {
            let dueClass = 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400';
            if (task.is_overdue) {
                dueClass = 'bg-red-100 dark:bg-red-950 text-red-600 dark:text-red-400 font-bold pulse-badge';
            } else if (task.relative_due === 'Today') {
                dueClass = 'bg-emerald-100 dark:bg-emerald-950/50 text-emerald-700 dark:text-emerald-400 font-semibold';
            } else if (task.relative_due === 'Tomorrow') {
                dueClass = 'bg-blue-100 dark:bg-blue-950/50 text-blue-700 dark:text-blue-400 font-semibold';
            }
            dueBadgeHtml = `
                <span class="inline-flex items-center px-2 py-0.5 rounded text-[11px] ${dueClass}">
                    📅 ${task.relative_due}
                </span>
            `;
        }

        // Tags badges
        const tagsHtml = task.tags.map(tag => `
            <span class="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium" style="background-color: ${tag.color}15; color: ${tag.color}">
                <span class="w-1.5 h-1.5 rounded-full mr-1" style="background-color: ${tag.color}"></span>
                ${tag.name}
            </span>
        `).join('');

        // Subtasks section
        let subtasksSectionHtml = '';
        if (task.subtasks && task.subtasks.length > 0) {
            const subtasksListHtml = task.subtasks.map(st => `
                <div class="flex items-center justify-between py-1 group/st">
                    <label class="flex items-center space-x-2 text-xs cursor-pointer select-none">
                        <input type="checkbox" ${st.is_completed ? 'checked' : ''} 
                            onchange="window.app.toggleSubtask(${st.id}, this.checked)"
                            class="w-3.5 h-3.5 text-indigo-600 rounded border-gray-300 dark:border-gray-600 focus:ring-0">
                        <span class="${st.is_completed ? 'line-through text-gray-400 dark:text-gray-500' : 'text-gray-700 dark:text-gray-300'}">
                            ${this.escapeHtml(st.title)}
                        </span>
                    </label>
                    <button onclick="window.app.deleteSubtask(${st.id})" class="opacity-0 group-hover/st:opacity-100 text-gray-400 hover:text-red-500 transition-opacity text-xs p-0.5">
                        ✕
                    </button>
                </div>
            `).join('');

            subtasksSectionHtml = `
                <div class="mt-3 pt-3 border-t border-gray-100 dark:border-gray-800">
                    <div class="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400 mb-1.5">
                        <span class="font-medium">Subtasks (${task.subtask_completed}/${task.subtask_total})</span>
                        <span class="font-mono">${task.progress_pct}%</span>
                    </div>
                    <div class="w-full bg-gray-100 dark:bg-gray-800 rounded-full h-1.5 mb-2 overflow-hidden">
                        <div class="bg-indigo-600 h-1.5 rounded-full transition-all duration-300" style="width: ${task.progress_pct}%"></div>
                    </div>
                    <div class="space-y-0.5 pl-1">
                        ${subtasksListHtml}
                    </div>
                </div>
            `;
        }

        return `
            <div class="task-card group relative bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-4 shadow-sm hover:shadow-md transition-all ${isCompleted ? 'opacity-70 bg-gray-50/70 dark:bg-gray-800/60' : ''}" 
                 draggable="true" 
                 data-task-id="${task.id}" 
                 data-task-position="${task.position}">
                
                <div class="flex items-start space-x-3">
                    <!-- Drag Handle -->
                    <button class="drag-handle cursor-grab active:cursor-grabbing text-gray-300 dark:text-gray-600 hover:text-gray-500 dark:hover:text-gray-400 mt-1 select-none" title="Drag to reorder">
                        ⋮⋮
                    </button>

                    <!-- Checkbox -->
                    <button onclick="window.app.toggleTask(${task.id}, event.clientX, event.clientY)" 
                            class="custom-checkbox w-5 h-5 rounded-full border-2 mt-0.5 flex items-center justify-center transition-all ${isCompleted ? 'bg-emerald-500 border-emerald-500 text-white' : 'border-gray-300 dark:border-gray-600 hover:border-emerald-500 text-transparent'}"
                            title="${isCompleted ? 'Mark active' : 'Mark completed'}">
                        <svg class="w-3 h-3 fill-current" viewBox="0 0 20 20">
                            <path d="M0 11l2-2 5 5L18 3l2 2L7 18z"/>
                        </svg>
                    </button>

                    <!-- Task Main Details -->
                    <div class="flex-1 min-w-0">
                        <div class="flex items-center justify-between">
                            <h4 class="text-sm font-semibold text-gray-900 dark:text-white truncate ${isCompleted ? 'line-through text-gray-400 dark:text-gray-500' : ''}">
                                ${this.escapeHtml(task.title)}
                            </h4>

                            <!-- Actions -->
                            <div class="flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                <button onclick="window.app.openEditTaskModal(${task.id})" class="p-1 text-gray-400 hover:text-indigo-600 rounded" title="Edit Task">
                                    ✎
                                </button>
                                <button onclick="window.app.deleteTask(${task.id}, '${this.escapeHtml(task.title)}')" class="p-1 text-gray-400 hover:text-red-600 rounded" title="Delete Task">
                                    🗑
                                </button>
                            </div>
                        </div>

                        ${task.description ? `
                            <p class="text-xs text-gray-600 dark:text-gray-400 mt-1 line-clamp-2">
                                ${this.escapeHtml(task.description)}
                            </p>
                        ` : ''}

                        <!-- Badges Row -->
                        <div class="flex flex-wrap items-center gap-1.5 mt-2.5">
                            <span class="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold border ${pConf.bg} ${pConf.text} ${pConf.border}">
                                ${pConf.label}
                            </span>
                            ${dueBadgeHtml}
                            ${tagsHtml}
                        </div>

                        ${subtasksSectionHtml}

                        <!-- Inline Add Subtask input -->
                        <div class="mt-2.5 pt-2 border-t border-dashed border-gray-100 dark:border-gray-700/60 hidden group-hover:block transition-all">
                            <form onsubmit="event.preventDefault(); const inp = this.querySelector('input'); window.app.addSubtask(${task.id}, inp.value); inp.value='';">
                                <input type="text" placeholder="+ Add quick subtask..." 
                                    class="w-full text-xs bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-indigo-500 text-gray-700 dark:text-gray-300">
                            </form>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    // ================== DRAG AND DROP ==================

    bindDragEvents() {
        const cards = document.querySelectorAll('.task-card');
        const container = document.getElementById('task-list-container');

        cards.forEach(card => {
            card.addEventListener('dragstart', (e) => {
                this.draggedTaskId = parseInt(card.dataset.taskId);
                card.classList.add('dragging');
                e.dataTransfer.effectAllowed = 'move';
                e.dataTransfer.setData('text/plain', card.dataset.taskId);
            });

            card.addEventListener('dragend', () => {
                card.classList.remove('dragging');
                cards.forEach(c => c.classList.remove('drag-over'));
            });

            card.addEventListener('dragover', (e) => {
                e.preventDefault();
                e.dataTransfer.dropEffect = 'move';
                if (!card.classList.contains('dragging')) {
                    card.classList.add('drag-over');
                }
            });

            card.addEventListener('dragleave', () => {
                card.classList.remove('drag-over');
            });

            card.addEventListener('drop', (e) => {
                e.preventDefault();
                card.classList.remove('drag-over');
                const targetTaskId = parseInt(card.dataset.taskId);
                if (this.draggedTaskId && this.draggedTaskId !== targetTaskId) {
                    this.handleCardDrop(this.draggedTaskId, targetTaskId);
                }
            });
        });
    }

    handleCardDrop(draggedId, targetId) {
        const draggedIndex = this.tasks.findIndex(t => t.id === draggedId);
        const targetIndex = this.tasks.findIndex(t => t.id === targetId);
        if (draggedIndex === -1 || targetIndex === -1) return;

        // Move item in memory
        const [movedTask] = this.tasks.splice(draggedIndex, 1);
        this.tasks.splice(targetIndex, 0, movedTask);

        // Reassign position values
        const orders = this.tasks.map((task, idx) => {
            task.position = idx;
            return { id: task.id, position: idx };
        });

        // Re-render immediate visual state
        this.renderTaskList();

        // Sync with backend
        this.reorderTasks(orders);
    }

    // ================== MODAL & FORM LOGIC ==================

    openCreateTaskModal() {
        this.editingTaskId = null;
        document.getElementById('task-modal-title').innerText = 'Create New Task';
        document.getElementById('task-title-input').value = '';
        document.getElementById('task-desc-input').value = '';
        document.getElementById('task-priority-input').value = 'P3';
        document.getElementById('task-due-input').value = '';
        document.getElementById('task-subtasks-input').value = '';

        // Uncheck all tags
        document.querySelectorAll('.task-modal-tag-checkbox').forEach(cb => cb.checked = false);

        this.openModal('task-modal');
        setTimeout(() => document.getElementById('task-title-input').focus(), 100);
    }

    openEditTaskModal(taskId) {
        const task = this.tasks.find(t => t.id === taskId);
        if (!task) return;

        this.editingTaskId = taskId;
        document.getElementById('task-modal-title').innerText = 'Edit Task';
        document.getElementById('task-title-input').value = task.title;
        document.getElementById('task-desc-input').value = task.description || '';
        document.getElementById('task-priority-input').value = task.priority || 'P3';
        document.getElementById('task-due-input').value = task.due_date ? task.due_date.substring(0, 10) : '';
        document.getElementById('task-subtasks-input').value = '';

        const activeTagIds = new Set(task.tags.map(t => t.id));
        document.querySelectorAll('.task-modal-tag-checkbox').forEach(cb => {
            cb.checked = activeTagIds.has(parseInt(cb.value));
        });

        this.openModal('task-modal');
    }

    async handleTaskModalSubmit(e) {
        e.preventDefault();
        const title = document.getElementById('task-title-input').value.trim();
        if (!title) return;

        const description = document.getElementById('task-desc-input').value.trim();
        const priority = document.getElementById('task-priority-input').value;
        const dueDate = document.getElementById('task-due-input').value || null;

        const selectedTagIds = Array.from(document.querySelectorAll('.task-modal-tag-checkbox:checked'))
            .map(cb => parseInt(cb.value));

        const subtaskLines = document.getElementById('task-subtasks-input').value
            .split('\n')
            .map(s => s.trim())
            .filter(s => s.length > 0);

        if (this.editingTaskId) {
            // Update existing
            await this.updateTask(this.editingTaskId, {
                title,
                description,
                priority,
                due_date: dueDate,
                tag_ids: selectedTagIds
            });
            // Add any newly typed subtasks
            for (const st of subtaskLines) {
                await this.addSubtask(this.editingTaskId, st);
            }
        } else {
            // Create new
            await this.createTask({
                title,
                description,
                priority,
                due_date: dueDate,
                tag_ids: selectedTagIds,
                subtasks: subtaskLines
            });
        }

        this.closeModal('task-modal');
    }

    openModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.remove('hidden');
            modal.classList.add('flex');
        }
    }

    closeModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.add('hidden');
            modal.classList.remove('flex');
        }
    }

    closeAllModals() {
        ['task-modal', 'command-palette-modal', 'analytics-modal', 'shortcuts-modal', 'tag-modal'].forEach(id => {
            this.closeModal(id);
        });
    }

    // ================== COMMAND PALETTE ==================

    openCommandPalette() {
        this.openModal('command-palette-modal');
        const input = document.getElementById('command-palette-input');
        if (input) {
            input.value = '';
            input.focus();
            this.filterCommandPalette('');
        }
    }

    filterCommandPalette(query) {
        const items = [
            { label: 'Create New Task', shortcut: 'N', action: () => { this.closeAllModals(); this.openCreateTaskModal(); } },
            { label: 'View Productivity Analytics', shortcut: 'A', action: () => { this.closeAllModals(); this.openModal('analytics-modal'); } },
            { label: 'Toggle Dark / Light Mode', shortcut: 'T', action: () => { this.toggleTheme(); } },
            { label: 'Show All Tasks', shortcut: '1', action: () => { this.setStatusFilter('all'); this.closeAllModals(); } },
            { label: 'Show Active Tasks', shortcut: '2', action: () => { this.setStatusFilter('active'); this.closeAllModals(); } },
            { label: 'Show Completed Tasks', shortcut: '3', action: () => { this.setStatusFilter('completed'); this.closeAllModals(); } },
            { label: 'Show Overdue Tasks', shortcut: '4', action: () => { this.setDueFilter('overdue'); this.closeAllModals(); } },
            { label: 'Show Today\'s Tasks', shortcut: '5', action: () => { this.setDueFilter('today'); this.closeAllModals(); } },
            { label: 'Show Tomorrow\'s Tasks', shortcut: '6', action: () => { this.setDueFilter('tomorrow'); this.closeAllModals(); } },
            { label: 'Show Keyboard Shortcuts', shortcut: '?', action: () => { this.closeAllModals(); this.openModal('shortcuts-modal'); } }
        ];

        const filtered = items.filter(it => it.label.toLowerCase().includes(query.toLowerCase()));
        const container = document.getElementById('command-palette-list');
        if (!container) return;

        if (filtered.length === 0) {
            container.innerHTML = `<div class="p-4 text-center text-xs text-gray-400">No matching commands found</div>`;
            return;
        }

        container.innerHTML = filtered.map((item, idx) => `
            <div onclick="window.app.executePaletteAction(${idx})" 
                 class="command-item flex items-center justify-between p-3 rounded-lg cursor-pointer text-sm text-gray-800 dark:text-gray-200">
                <span>${item.label}</span>
                <kbd class="px-2 py-0.5 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded text-xs font-mono text-gray-500">${item.shortcut}</kbd>
            </div>
        `).join('');

        this.paletteFilteredItems = filtered;
    }

    executePaletteAction(idx) {
        if (this.paletteFilteredItems && this.paletteFilteredItems[idx]) {
            this.paletteFilteredItems[idx].action();
        }
    }

    // ================== FILTERS & SORTING ==================

    setStatusFilter(status) {
        this.filterStatus = status;
        this.filterDue = null;
        this.updateFilterButtons();
        this.loadTasks();
    }

    setDueFilter(due) {
        this.filterDue = due;
        this.filterStatus = 'all';
        this.updateFilterButtons();
        this.loadTasks();
    }

    setPriorityFilter(priority) {
        this.filterPriority = this.filterPriority === priority ? null : priority;
        this.updateFilterButtons();
        this.loadTasks();
    }

    setTagFilter(tagId) {
        this.filterTagId = this.filterTagId === tagId ? null : tagId;
        this.renderTagList();
        this.loadTasks();
    }

    setSort(by) {
        if (this.sortBy === by) {
            this.sortOrder = this.sortOrder === 'asc' ? 'desc' : 'asc';
        } else {
            this.sortBy = by;
            this.sortOrder = 'asc';
        }
        this.loadTasks();
    }

    updateFilterButtons() {
        document.querySelectorAll('[data-filter-btn]').forEach(btn => {
            const f = btn.dataset.filterBtn;
            if (f === this.filterStatus && !this.filterDue) {
                btn.className = 'w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-semibold bg-indigo-50 dark:bg-indigo-950/40 text-indigo-700 dark:text-indigo-300';
            } else {
                btn.className = 'w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800';
            }
        });

        document.querySelectorAll('[data-due-btn]').forEach(btn => {
            const d = btn.dataset.dueBtn;
            if (d === this.filterDue) {
                btn.className = 'w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-semibold bg-indigo-50 dark:bg-indigo-950/40 text-indigo-700 dark:text-indigo-300';
            } else {
                btn.className = 'w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800';
            }
        });

        document.querySelectorAll('[data-priority-btn]').forEach(btn => {
            const p = btn.dataset.priorityBtn;
            if (p === this.filterPriority) {
                btn.classList.add('ring-2', 'ring-offset-1', 'ring-indigo-500');
            } else {
                btn.classList.remove('ring-2', 'ring-offset-1', 'ring-indigo-500');
            }
        });
    }

    // ================== SHORTCUTS & EVENT LISTENERS ==================

    bindGlobalShortcuts() {
        window.addEventListener('keydown', (e) => {
            // Ignore keyboard shortcuts when typing in inputs/textareas
            const activeTag = document.activeElement.tagName.toLowerCase();
            const isTyping = (activeTag === 'input' || activeTag === 'textarea');

            // Cmd/Ctrl + K: Toggle Command Palette
            if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
                e.preventDefault();
                this.openCommandPalette();
                return;
            }

            // Escape: Close all modals
            if (e.key === 'Escape') {
                this.closeAllModals();
                return;
            }

            if (!isTyping) {
                if (e.key === 'n' || e.key === 'N') {
                    e.preventDefault();
                    this.openCreateTaskModal();
                } else if (e.key === '/') {
                    e.preventDefault();
                    const s = document.getElementById('global-search-input');
                    if (s) s.focus();
                } else if (e.key === 't' || e.key === 'T') {
                    e.preventDefault();
                    this.toggleTheme();
                } else if (e.key === '?') {
                    e.preventDefault();
                    this.openModal('shortcuts-modal');
                }
            }
        });
    }

    bindUIEvents() {
        // Search Input
        const searchInput = document.getElementById('global-search-input');
        if (searchInput) {
            let timeout = null;
            searchInput.addEventListener('input', (e) => {
                clearTimeout(timeout);
                timeout = setTimeout(() => {
                    this.searchQuery = e.target.value.trim();
                    this.loadTasks();
                }, 250);
            });
        }

        // Quick Add Form
        const quickForm = document.getElementById('quick-add-task-form');
        if (quickForm) {
            quickForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const inp = document.getElementById('quick-add-input');
                const title = inp.value.trim();
                if (!title) return;
                inp.value = '';
                await this.createTask({ title, priority: 'P3' });
            });
        }

        // Task Form
        const taskForm = document.getElementById('task-modal-form');
        if (taskForm) {
            taskForm.addEventListener('submit', (e) => this.handleTaskModalSubmit(e));
        }

        // Tag Form
        const tagForm = document.getElementById('tag-modal-form');
        if (tagForm) {
            tagForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const name = document.getElementById('tag-name-input').value;
                const color = document.getElementById('tag-color-input').value;
                await this.createTag(name, color);
                this.closeModal('tag-modal');
            });
        }

        // Palette Search
        const paletteSearch = document.getElementById('command-palette-input');
        if (paletteSearch) {
            paletteSearch.addEventListener('input', (e) => {
                this.filterCommandPalette(e.target.value);
            });
        }
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const toast = document.createElement('div');
        const bgColors = {
            info: 'bg-gray-900 text-white dark:bg-gray-100 dark:text-gray-900',
            success: 'bg-emerald-600 text-white',
            error: 'bg-red-600 text-white'
        };

        toast.className = `${bgColors[type] || bgColors.info} px-4 py-2.5 rounded-xl shadow-lg text-xs font-medium flex items-center space-x-2 transition-all duration-300 transform translate-y-4 opacity-0`;
        toast.innerHTML = `<span>${message}</span>`;
        container.appendChild(toast);

        // Fade in
        requestAnimationFrame(() => {
            toast.classList.remove('translate-y-4', 'opacity-0');
        });

        // Remove after 3s
        setTimeout(() => {
            toast.classList.add('opacity-0', 'translate-y-2');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    escapeHtml(str) {
        if (!str) return '';
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.app = new TodoApp();
});
