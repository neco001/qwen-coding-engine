"""AI Enhancement Layer for dependency analysis.

Provides AI-powered suggestions and risk scoring for code changes.
"""

from typing import Dict, List, Any, Optional


class AIEnhancer:
    """AI-powered enhancement layer for dependency analysis."""
    
    async def enhance_dependencies(
        self, static_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance static dependency analysis with AI suggestions.
        
        Args:
            static_analysis: Result from DependencyTracker.analyze_project()
            
        Returns:
            Enhanced analysis with suggestions
        """
        result = dict(static_analysis)
        suggestions = []
        
        # Analyze dependency patterns
        dependencies = static_analysis.get("dependencies", [])
        files = static_analysis.get("files", {})
        
        # Detect tight coupling (files with many dependencies)
        dependency_counts = {}
        for dep in dependencies:
            from_file = dep.get("from_file", "")
            dependency_counts[from_file] = dependency_counts.get(from_file, 0) + 1
        
        for file_path, count in dependency_counts.items():
            if count >= 4:
                suggestions.append({
                    "type": "tight_coupling",
                    "file": file_path,
                    "message": f"File {file_path} has {count} dependencies - consider refactoring",
                    "severity": "warning" if count < 8 else "critical",
                })
        
        # Detect potential circular dependencies
        dep_graph = {}
        for dep in dependencies:
            from_file = dep.get("from_file", "")
            to_file = dep.get("to_file", "")
            if from_file not in dep_graph:
                dep_graph[from_file] = set()
            dep_graph[from_file].add(to_file)
        
        # Simple cycle detection
        for file_a, deps_a in dep_graph.items():
            for file_b in deps_a:
                if file_b in dep_graph and file_a in dep_graph[file_b]:
                    suggestions.append({
                        "type": "circular_dependency",
                        "files": [file_a, file_b],
                        "message": f"Circular dependency detected between {file_a} and {file_b}",
                        "severity": "warning",
                    })
        
        # Detect files with no dependencies (potential dead code)
        if files:
            all_targets = set()
            for dep in dependencies:
                all_targets.add(dep.get("to_file", ""))
            
            for file_path in files.keys():
                if file_path not in all_targets and file_path != "main.py":
                    # Check if it's not an entry point
                    file_info = files[file_path]
                    if not file_info.get("imports"):
                        suggestions.append({
                            "type": "potential_dead_code",
                            "file": file_path,
                            "message": f"File {file_path} is not imported by any other file",
                            "severity": "info",
                        })
        
        result["suggestions"] = suggestions
        return result
    
    async def get_risk_score(self, change_info: Dict[str, Any]) -> float:
        """Calculate risk score for a proposed code change.
        
        Args:
            change_info: Dict with keys:
                - modified_files: List of file paths
                - lines_changed: Number of lines changed
                - dependencies_affected: Number of dependencies affected
                
        Returns:
            Risk score between 0.0 (low risk) and 1.0 (high risk)
        """
        modified_files = change_info.get("modified_files", [])
        lines_changed = change_info.get("lines_changed", 0)
        dependencies_affected = change_info.get("dependencies_affected", 0)
        
        # Base score from lines changed
        lines_score = min(lines_changed / 500.0, 1.0)
        
        # Score from number of files
        files_score = min(len(modified_files) / 10.0, 1.0)
        
        # Score from dependencies affected
        deps_score = min(dependencies_affected / 20.0, 1.0)
        
        # Weighted average
        risk_score = (
            0.3 * lines_score +
            0.3 * files_score +
            0.4 * deps_score
        )
        
        return min(max(risk_score, 0.0), 1.0)
    
    async def suggest_validator_triggers(
        self, change_info: Dict[str, Any]
    ) -> List[str]:
        """Suggest which validators should be triggered for a change.
        
        Args:
            change_info: Same as get_risk_score()
            
        Returns:
            List of validator trigger names
        """
        triggers = []
        risk_score = await self.get_risk_score(change_info)
        
        # High risk changes trigger more validators
        if risk_score >= 0.7:
            triggers.extend(["full_test_suite", "integration_tests", "lint", "type_check"])
        elif risk_score >= 0.4:
            triggers.extend(["affected_tests", "lint"])
        else:
            triggers.append("lint")
        
        # Check for specific patterns
        modified_files = change_info.get("modified_files", [])
        for file_path in modified_files:
            if "test" in file_path:
                if "test_suite" not in triggers:
                    triggers.append("affected_tests")
            if ".pyi" in file_path or "typing" in file_path:
                if "type_check" not in triggers:
                    triggers.append("type_check")
        
        return list(set(triggers))
