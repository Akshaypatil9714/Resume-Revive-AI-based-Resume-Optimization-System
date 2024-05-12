from django.utils.deprecation import MiddlewareMixin

class DisableXFrameOptionsMiddleware(MiddlewareMixin):
    def process_response(self, request, response):
        if request.path.startswith('/media/'):
            response['X-Frame-Options'] = 'SAMEORIGIN'
        return response