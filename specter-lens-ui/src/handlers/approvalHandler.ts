// specter-lens-ui/src/handlers/approvalHandler.ts

interface ChangeRequest {
  id: string;
  description?: string;
  // Add other relevant fields as needed
}

interface ApprovalRequest {
  changes: ChangeRequest[];
  action: 'approve' | 'reject';
  reason?: string;
}

interface ApprovalResult {
  changeId: string;
  status: 'approved' | 'rejected' | 'failed';
  message?: string;
}

interface ApprovalResponse {
  success: boolean;
  results: ApprovalResult[];
}

class ApprovalHandler {
  private readonly baseUrl: string = 'http://localhost:8878';
  private readonly defaultTimeout: number = 30000; // 30 seconds
  private readonly maxRetries: number = 2;
  private readonly retryDelay: number = 1000; // 1 second

  /**
   * Send batch approval request to MCP server
   * @param request - Approval request containing changes and action
   * @param timeout - Request timeout in milliseconds (default: 30s)
   * @returns Promise with approval results
   */
  async handleBatchApproval(
    request: ApprovalRequest,
    timeout: number = this.defaultTimeout
  ): Promise<ApprovalResponse> {
    const url = `${this.baseUrl}/api/approval`;
    
    // Validate input
    if (!request.changes || request.changes.length === 0) {
      return {
        success: false,
        results: []
      };
    }

    // Build request payload
    const payload: ApprovalRequest = {
      changes: request.changes,
      action: request.action,
      reason: request.reason
    };

    // Retry logic
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      try {
        const response = await this.sendRequestWithTimeout(url, payload, timeout);
        
        // Validate response structure
        if (!response || typeof response.success !== 'boolean' || !Array.isArray(response.results)) {
          throw new Error('Invalid response format from MCP server');
        }

        return response;
      } catch (error) {
        // If this is the last attempt, throw the error
        if (attempt === this.maxRetries) {
          console.error(`Batch approval failed after ${this.maxRetries + 1} attempts:`, error);
          
          // Create failure results for all changes
          const failureResults = request.changes.map(change => ({
            changeId: change.id,
            status: 'failed',
            message: error instanceof Error ? error.message : 'Unknown error occurred'
          }));
          
          return {
            success: false,
            results: failureResults
          };
        }
        
        // Wait before retrying
        await new Promise(resolve => setTimeout(resolve, this.retryDelay * (attempt + 1)));
      }
    }

    // Fallback (should never reach here due to throw in loop)
    return {
      success: false,
      results: request.changes.map(change => ({
        changeId: change.id,
        status: 'failed',
        message: 'Unexpected error in approval handler'
      }))
    };
  }

  /**
   * Send HTTP request with timeout support
   */
  private async sendRequestWithTimeout(
    url: string,
    payload: ApprovalRequest,
    timeout: number
  ): Promise<ApprovalResponse> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text().catch(() => '');
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
      }

      const result: ApprovalResponse = await response.json();
      return result;
    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Request timeout');
      }
      
      throw error;
    }
  }
}

// Export singleton instance
export const approvalHandler = new ApprovalHandler();
