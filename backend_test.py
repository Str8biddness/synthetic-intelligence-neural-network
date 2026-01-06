#!/usr/bin/env python3
"""
Comprehensive Backend API Tests for Synthetic Intelligence Engine
Tests all SI endpoints for functionality, performance, and response format
"""

import requests
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Any

class SIEngineAPITester:
    def __init__(self, base_url="https://patternmind.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.response_times = []
        self.test_results = []

    def log_test(self, name: str, success: bool, details: str = "", response_time: float = 0):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
        
        result = {
            "test_name": name,
            "success": success,
            "details": details,
            "response_time_ms": response_time,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {name}")
        if details:
            print(f"    {details}")
        if response_time > 0:
            print(f"    Response time: {response_time:.1f}ms")
        print()

    def make_request(self, method: str, endpoint: str, data: Dict = None, expected_status: int = 200) -> tuple:
        """Make HTTP request and measure response time"""
        url = f"{self.api_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        start_time = time.time()
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")
                
            response_time = (time.time() - start_time) * 1000
            self.response_times.append(response_time)
            
            success = response.status_code == expected_status
            response_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
            
            return success, response_data, response_time, response.status_code
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return False, {}, response_time, 0

    def test_health_check(self):
        """Test health check endpoint"""
        success, data, response_time, status_code = self.make_request('GET', 'health')
        
        if success and data.get('status') == 'healthy':
            details = f"SI engine operational: {data.get('si_engine', 'unknown')}"
            self.log_test("Health Check", True, details, response_time)
            return True
        else:
            self.log_test("Health Check", False, f"Status code: {status_code}", response_time)
            return False

    def test_si_ask_endpoint(self):
        """Test main SI query endpoint"""
        test_queries = [
            "What is consciousness?",
            "Explain quantum mechanics",
            "How does gravity work?",
            "What is the meaning of life?"
        ]
        
        all_passed = True
        for query in test_queries:
            success, data, response_time, status_code = self.make_request(
                'POST', 'si/ask', 
                {'query': query}
            )
            
            if success:
                # Validate response structure
                required_fields = ['id', 'query', 'response', 'confidence', 'reasoning_strategy', 
                                 'patterns_used', 'domains_involved', 'response_time_ms']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test(f"SI Ask - {query[:20]}...", False, 
                                f"Missing fields: {missing_fields}", response_time)
                    all_passed = False
                else:
                    # Check response time requirement (sub-500ms)
                    actual_response_time = data.get('response_time_ms', 0)
                    sub_500ms = actual_response_time < 500
                    
                    confidence = data.get('confidence', 0)
                    patterns_used = data.get('patterns_used', 0)
                    
                    details = f"Confidence: {confidence:.2f}, Patterns: {patterns_used}, " \
                             f"Internal time: {actual_response_time:.1f}ms, Sub-500ms: {sub_500ms}"
                    
                    self.log_test(f"SI Ask - {query[:20]}...", True, details, response_time)
            else:
                self.log_test(f"SI Ask - {query[:20]}...", False, 
                            f"Status code: {status_code}", response_time)
                all_passed = False
        
        return all_passed

    def test_si_simulate_endpoint(self):
        """Test parallel reality simulation endpoint"""
        test_data = {
            'query': 'What if gravity was twice as strong?',
            'assumptions': ['Earth mass doubled', 'Same atmospheric composition']
        }
        
        success, data, response_time, status_code = self.make_request(
            'POST', 'si/simulate', test_data
        )
        
        if success and 'branches' in data:
            branches = data.get('branches', [])
            details = f"Generated {len(branches)} reality branches"
            self.log_test("SI Simulate", True, details, response_time)
            return True
        else:
            self.log_test("SI Simulate", False, f"Status code: {status_code}", response_time)
            return False

    def test_si_causal_endpoint(self):
        """Test causal reasoning endpoint"""
        test_data = {'query': 'What causes climate change?'}
        
        success, data, response_time, status_code = self.make_request(
            'POST', 'si/causal', test_data
        )
        
        if success:
            details = f"Causal analysis completed"
            self.log_test("SI Causal Reasoning", True, details, response_time)
            return True
        else:
            self.log_test("SI Causal Reasoning", False, f"Status code: {status_code}", response_time)
            return False

    def test_si_stats_endpoint(self):
        """Test statistics endpoint"""
        success, data, response_time, status_code = self.make_request('GET', 'si/stats')
        
        if success:
            patterns_total = data.get('patterns', {}).get('total', 0)
            entities_total = data.get('entities', {}).get('total', 0)
            performance = data.get('performance', {})
            
            details = f"Patterns: {patterns_total}, Entities: {entities_total}, " \
                     f"Avg response: {performance.get('avg_response_time_ms', 0):.1f}ms"
            
            self.log_test("SI Statistics", True, details, response_time)
            return True
        else:
            self.log_test("SI Statistics", False, f"Status code: {status_code}", response_time)
            return False

    def test_si_patterns_endpoint(self):
        """Test patterns endpoint"""
        success, data, response_time, status_code = self.make_request('GET', 'si/patterns')
        
        if success and 'patterns' in data:
            pattern_count = data.get('count', 0)
            patterns = data.get('patterns', [])
            
            details = f"Retrieved {pattern_count} patterns, showing {len(patterns)}"
            self.log_test("SI Patterns", True, details, response_time)
            return True
        else:
            self.log_test("SI Patterns", False, f"Status code: {status_code}", response_time)
            return False

    def test_si_self_observe_endpoint(self):
        """Test self-observation endpoint"""
        success, data, response_time, status_code = self.make_request('GET', 'si/self-observe')
        
        if success:
            details = "Self-observation completed"
            self.log_test("SI Self-Observe", True, details, response_time)
            return True
        else:
            self.log_test("SI Self-Observe", False, f"Status code: {status_code}", response_time)
            return False

    def test_generate_image_endpoint(self):
        """Test image generation endpoint"""
        test_descriptions = [
            "sunset over ocean with sailboat",
            "person walking dog in park", 
            "house with tree on sunny day",
            "red car on mountain road"
        ]
        
        all_passed = True
        for description in test_descriptions:
            success, data, response_time, status_code = self.make_request(
                'POST', 'generate-image',
                {'description': description, 'use_optimizer': True}
            )
            
            if success:
                # Validate response structure
                required_fields = ['success', 'svg', 'total_time_ms']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test(f"Generate Image - {description[:20]}...", False,
                                f"Missing fields: {missing_fields}", response_time)
                    all_passed = False
                elif not data.get('success'):
                    self.log_test(f"Generate Image - {description[:20]}...", False,
                                f"Generation failed: {data.get('error', 'Unknown error')}", response_time)
                    all_passed = False
                else:
                    # Check sub-500ms requirement
                    generation_time = data.get('total_time_ms', 0)
                    sub_500ms = generation_time < 500
                    svg_length = len(data.get('svg', ''))
                    
                    details = f"Generated SVG ({svg_length} chars), Time: {generation_time:.1f}ms, Sub-500ms: {sub_500ms}"
                    self.log_test(f"Generate Image - {description[:20]}...", True, details, response_time)
            else:
                self.log_test(f"Generate Image - {description[:20]}...", False,
                            f"Status code: {status_code}", response_time)
                all_passed = False
        
        return all_passed

    def test_visual_patterns_endpoint(self):
        """Test visual patterns endpoint"""
        success, data, response_time, status_code = self.make_request('GET', 'visual-patterns')
        
        if success and 'patterns' in data:
            pattern_count = data.get('count', 0)
            patterns = data.get('patterns', [])
            
            # Check if we have expected base patterns
            pattern_names = [p.get('name', '') for p in patterns]
            expected_patterns = ['circle', 'rectangle', 'triangle', 'tree', 'house', 'car', 'sun', 'cloud']
            found_patterns = [name for name in expected_patterns if name in pattern_names]
            
            details = f"Retrieved {pattern_count} patterns, found {len(found_patterns)}/{len(expected_patterns)} base patterns"
            self.log_test("Visual Patterns", True, details, response_time)
            return True
        else:
            self.log_test("Visual Patterns", False, f"Status code: {status_code}", response_time)
            return False

    def test_image_generation_stats_endpoint(self):
        """Test image generation statistics endpoint"""
        success, data, response_time, status_code = self.make_request('GET', 'image-generation/stats')
        
        if success:
            total_generations = data.get('total_generations', 0)
            avg_time = data.get('avg_generation_time_ms', 0)
            cache_hit_rate = data.get('cache_hit_rate', 0)
            
            details = f"Total generations: {total_generations}, Avg time: {avg_time:.1f}ms, Cache hit rate: {cache_hit_rate:.1%}"
            self.log_test("Image Generation Stats", True, details, response_time)
            return True
        else:
            self.log_test("Image Generation Stats", False, f"Status code: {status_code}", response_time)
            return False

    def test_performance_requirements(self):
        """Test sub-500ms response time requirement"""
        if not self.response_times:
            self.log_test("Performance Check", False, "No response times recorded")
            return False
        
        avg_response_time = sum(self.response_times) / len(self.response_times)
        max_response_time = max(self.response_times)
        sub_500ms_count = sum(1 for rt in self.response_times if rt < 500)
        sub_500ms_percentage = (sub_500ms_count / len(self.response_times)) * 100
        
        details = f"Avg: {avg_response_time:.1f}ms, Max: {max_response_time:.1f}ms, " \
                 f"Sub-500ms: {sub_500ms_percentage:.1f}% ({sub_500ms_count}/{len(self.response_times)})"
        
        # Consider it a pass if at least 80% of requests are sub-500ms
        success = sub_500ms_percentage >= 80
        self.log_test("Performance Requirements", success, details)
        return success

    def run_all_tests(self):
        """Run all tests"""
        print("🧠 Starting Synthetic Intelligence Engine API Tests")
        print("=" * 60)
        print()
        
        # Test basic connectivity first
        if not self.test_health_check():
            print("❌ Health check failed - stopping tests")
            return False
        
        # Test all SI endpoints
        tests = [
            self.test_si_ask_endpoint,
            self.test_si_simulate_endpoint,
            self.test_si_causal_endpoint,
            self.test_si_stats_endpoint,
            self.test_si_patterns_endpoint,
            self.test_si_self_observe_endpoint,
            self.test_generate_image_endpoint,
            self.test_visual_patterns_endpoint,
            self.test_image_generation_stats_endpoint,
            self.test_performance_requirements
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                self.log_test(test.__name__, False, f"Exception: {str(e)}")
        
        # Print summary
        print("=" * 60)
        print(f"📊 Test Summary: {self.tests_passed}/{self.tests_run} tests passed")
        
        if self.response_times:
            avg_time = sum(self.response_times) / len(self.response_times)
            print(f"⚡ Average response time: {avg_time:.1f}ms")
        
        success_rate = (self.tests_passed / self.tests_run) * 100 if self.tests_run > 0 else 0
        print(f"✅ Success rate: {success_rate:.1f}%")
        print()
        
        return self.tests_passed == self.tests_run

    def get_test_summary(self) -> Dict[str, Any]:
        """Get test summary for reporting"""
        return {
            "total_tests": self.tests_run,
            "passed_tests": self.tests_passed,
            "failed_tests": self.tests_run - self.tests_passed,
            "success_rate": (self.tests_passed / self.tests_run) * 100 if self.tests_run > 0 else 0,
            "avg_response_time_ms": sum(self.response_times) / len(self.response_times) if self.response_times else 0,
            "max_response_time_ms": max(self.response_times) if self.response_times else 0,
            "sub_500ms_percentage": (sum(1 for rt in self.response_times if rt < 500) / len(self.response_times)) * 100 if self.response_times else 0,
            "test_results": self.test_results
        }

def main():
    """Main test execution"""
    tester = SIEngineAPITester()
    
    try:
        success = tester.run_all_tests()
        
        # Save detailed results
        summary = tester.get_test_summary()
        with open('/app/backend_test_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test execution failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())