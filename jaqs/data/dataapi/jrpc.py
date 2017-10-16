import threading
import _jrpc
import traceback

def set_log_dir(log_path):
    _jrpc.set_log_dir(log_path)

class JRpcWrap :
    
    def __init__(self) :
        self._handle = 0
        self._callback = None
        
        
    def __del__(self):
        self.close()

    def connect(self, addr):
        """
        Connect to remote server using addr. Return True if success else False
        """
        
        #There might have some incoming events in _jrpc.connect. In order not to
        # lose them, the _my_callback should be passed to _jrpc.

        self._handle = _jrpc.connect(addr, self._my_callback)
        return True if self._handle else False

    def close(self):
        _jrpc.close(self._handle)
        self._handle = 0

    def _my_callback(self, event, data):
        if self._callback :
            self._callback(event, data)

    def set_callback(self, callback):
        """
        Notification callback, prototype (event, data)
        event: 
                ".sys.connected",
                ".sys.disconnected"
                and others
        """
        self._callback = callback

    def call(self, method, param, callback, timeout=6000):
        """
        Async call. When the client receives result or timeout, callback will be called.
        Callback(call_id: Int, method: String, result: Any, error: [code, msg])
            if result is None, means call faiure

        return call_id: Int, zero means error.
        """
        assert self._handle, "none handle"
        return _jrpc.call(self._handle, method, param, callback, timeout)

class JRpcClient:
    """
    Simulate behaviour of jsonrpc.JsonRpcClient so AdminApi,TradeApi can be simple.
    """
    def __init__(self):
        self._jrpc_client = JRpcWrap()
        self.on_rpc_callback = None
        self.on_disconnected = None
        self.on_connected    = None

        self._jrpc_client.set_callback(self._my_callback)

    def __del__(self):
        self.close()

    def connect(self, addr):
        """
        return True or False
        """
        return self._jrpc_client.connect(addr)

    def close(self):
        if self._jrpc_client :
            self._jrpc_client.close()
            self._jrpc_client = None

    def _my_callback(self, method, data):
        if   method == ".sys.connected" :
            if self.on_connected : self.on_connected()

        elif method == ".sys.disconnected" :
            if self.on_disconnected : self.on_disconnected()

        else:
             if self.on_rpc_callback : self.on_rpc_callback(method, data)


    def set_heartbeat_options(self, interval, timeout):
        # self._heartbeat_interval = interval
        # self._heartbeat_timeout = timeout
        pass

    def call(self, method, params, timeout = 6) :
        if not timeout :
            try :
                self._jrpc_client.call(method, params, lambda id,m,r,e: None, timeout*1000)
                return {'result': True}
            except Exception as e:
                traceback.print_exc()
                return {'error': {'error': -1, 'message' : str(e)}}
        else:
            ret = {}
            cv = threading.Condition()
            try:
                def _lambda(id, m, r, e):
                    cv.acquire()
                    ret['result'] = r
                    ret['error']  = e
                    cv.notify()
                    cv.release()

                call_id = self._jrpc_client.call(method, params, _lambda, timeout*1000)
                if not call_id:
                    # if call_id is None, an exception should have been raised.
                    return {'error': {'error': -1, 'message' : 'unknown error'} }

            except Exception as e:
                traceback.print_exc()
                ret = {'error': {'error': -1, 'message' : str(e)}}
                return ret

            try:
                cv.acquire()
                if not ret:
                    # cv.wait a little longer than jrpc.call
                    cv.wait(timeout + 0.5)

                if not ret:
                    ret['error'] = {'error': -1, 'message' : 'timeout'}

            except Exception as e:
                traceback.print_exc()
                ret = {'error': {'error': -1, 'message' : str(e)}}
                    
            finally:
                cv.release()

            return ret

