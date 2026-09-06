// Productivity Analytics Visualization Engine
class AnalyticsRenderer {
    static renderScoreGauge(score, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        let statusText = "Getting Started";
        let statusColor = "text-amber-500";
        if (score >= 80) {
            statusText = "Peak Performer 🔥";
            statusColor = "text-emerald-500";
        } else if (score >= 60) {
            statusText = "High Momentum ⚡";
            statusColor = "text-indigo-500";
        } else if (score >= 40) {
            statusText = "Steady Flow 🎯";
            statusColor = "text-blue-500";
        }

        const radius = 40;
        const circumference = 2 * Math.PI * radius;
        const offset = circumference - (score / 100) * circumference;

        container.innerHTML = `
            <div class="flex items-center space-x-4">
                <div class="relative w-24 h-24 flex items-center justify-center">
                    <svg class="w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
                        <circle cx="50" cy="50" r="${radius}" stroke="currentColor" stroke-width="8" class="text-gray-200 dark:text-gray-700" fill="transparent" />
                        <circle cx="50" cy="50" r="${radius}" stroke="#6366f1" stroke-width="8" stroke-dasharray="${circumference}" stroke-dashoffset="${offset}" stroke-linecap="round" fill="transparent" class="transition-all duration-1000 ease-out" />
                    </svg>
                    <div class="absolute flex flex-col items-center justify-center">
                        <span class="text-2xl font-black text-gray-900 dark:text-white">${score}</span>
                        <span class="text-[10px] font-bold text-gray-400 uppercase tracking-wider">Score</span>
                    </div>
                </div>
                <div>
                    <div class="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">Productivity Level</div>
                    <div class="text-base font-bold ${statusColor}">${statusText}</div>
                    <div class="text-xs text-gray-500 dark:text-gray-400 mt-0.5">Based on completion rate, streaks & on-time tasks</div>
                </div>
            </div>
        `;
    }

    static renderVelocityChart(velocityData, containerId) {
        const container = document.getElementById(containerId);
        if (!container || !velocityData || velocityData.length === 0) return;

        const maxVal = Math.max(1, ...velocityData.map(d => Math.max(d.completed_count, d.created_count)));

        let barsHtml = velocityData.map(d => {
            const completedHeight = Math.round((d.completed_count / maxVal) * 90);
            const createdHeight = Math.round((d.created_count / maxVal) * 90);

            return `
                <div class="flex-1 flex flex-col items-center group relative h-36 justify-end">
                    <!-- Tooltip -->
                    <div class="opacity-0 group-hover:opacity-100 transition-opacity absolute -top-12 bg-gray-900 dark:bg-gray-800 text-white text-[11px] rounded px-2 py-1 pointer-events-none whitespace-nowrap shadow-lg z-20">
                        <div class="font-bold">${d.date} (${d.day_name})</div>
                        <div class="text-emerald-400">✓ ${d.completed_count} completed</div>
                        <div class="text-indigo-400">+ ${d.created_count} created</div>
                    </div>

                    <!-- Bars -->
                    <div class="w-full flex items-end justify-center space-x-1 h-28">
                        <div class="w-3 bg-emerald-500 hover:bg-emerald-400 rounded-t transition-all duration-300" style="height: ${Math.max(4, completedHeight)}%" title="${d.completed_count} completed"></div>
                        <div class="w-3 bg-indigo-500 hover:bg-indigo-400 rounded-t transition-all duration-300 opacity-75" style="height: ${Math.max(4, createdHeight)}%" title="${d.created_count} created"></div>
                    </div>

                    <!-- Day Label -->
                    <span class="text-[11px] font-medium text-gray-500 dark:text-gray-400 mt-2">${d.day_name}</span>
                </div>
            `;
        }).join('');

        container.innerHTML = `
            <div class="flex items-center justify-between mb-3 text-xs text-gray-500 dark:text-gray-400">
                <div class="flex items-center space-x-3">
                    <span class="flex items-center"><span class="w-2.5 h-2.5 bg-emerald-500 rounded-full mr-1.5"></span>Completed</span>
                    <span class="flex items-center"><span class="w-2.5 h-2.5 bg-indigo-500 rounded-full mr-1.5"></span>Created</span>
                </div>
                <span>Last 7 Days</span>
            </div>
            <div class="flex items-end justify-between space-x-2 pt-2 border-b border-gray-200 dark:border-gray-700">
                ${barsHtml}
            </div>
        `;
    }

    static renderTagBreakdown(tagData, containerId) {
        const container = document.getElementById(containerId);
        if (!container || !tagData) return;

        const totalTagged = tagData.reduce((acc, t) => acc + t.count, 0);

        if (totalTagged === 0) {
            container.innerHTML = `<div class="text-xs text-gray-400 py-3 text-center">No tagged tasks yet</div>`;
            return;
        }

        const segmentsHtml = tagData.filter(t => t.count > 0).map(t => {
            const pct = Math.round((t.count / totalTagged) * 100);
            return `<div style="width: ${pct}%; background-color: ${t.color || '#6366f1'}" class="h-3 first:rounded-l-full last:rounded-r-full transition-all duration-500" title="${t.name}: ${t.count} (${pct}%)"></div>`;
        }).join('');

        const legendHtml = tagData.filter(t => t.count > 0).map(t => {
            const pct = Math.round((t.count / totalTagged) * 100);
            return `
                <div class="flex items-center justify-between text-xs py-1">
                    <div class="flex items-center space-x-2">
                        <span class="w-2.5 h-2.5 rounded-full" style="background-color: ${t.color || '#6366f1'}"></span>
                        <span class="font-medium text-gray-700 dark:text-gray-300">${t.name}</span>
                    </div>
                    <span class="text-gray-500 dark:text-gray-400 font-mono">${t.count} (${pct}%)</span>
                </div>
            `;
        }).join('');

        container.innerHTML = `
            <div class="w-full bg-gray-100 dark:bg-gray-700 rounded-full flex overflow-hidden mb-3">
                ${segmentsHtml}
            </div>
            <div class="space-y-0.5">
                ${legendHtml}
            </div>
        `;
    }
}

window.AnalyticsRenderer = AnalyticsRenderer;
