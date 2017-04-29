/* soapDAMISService.cpp
   Generated by gSOAP 2.8.14 from DAMIS.h

Copyright(C) 2000-2013, Robert van Engelen, Genivia Inc. All Rights Reserved.
The generated code is released under ONE of the following licenses:
GPL or Genivia's license for commercial use.
This program is released under the GPL with the additional exemption that
compiling, linking, and/or using OpenSSL is allowed.
*/

#include "soapDAMISService.h"

DAMISService::DAMISService()
{	this->soap = soap_new();
	this->own = true;
	DAMISService_init(SOAP_IO_DEFAULT, SOAP_IO_DEFAULT);
}

DAMISService::DAMISService(struct soap *_soap)
{	this->soap = _soap;
	this->own = false;
	DAMISService_init(_soap->imode, _soap->omode);
}

DAMISService::DAMISService(soap_mode iomode)
{	this->soap = soap_new();
	this->own = true;
	DAMISService_init(iomode, iomode);
}

DAMISService::DAMISService(soap_mode imode, soap_mode omode)
{	this->soap = soap_new();
	this->own = true;
	DAMISService_init(imode, omode);
}

DAMISService::~DAMISService()
{	if (this->own)
		soap_free(this->soap);
}

void DAMISService::DAMISService_init(soap_mode imode, soap_mode omode)
{	soap_imode(this->soap, imode);
	soap_omode(this->soap, omode);
	static const struct Namespace namespaces[] =
{
	{"SOAP-ENV", "http://schemas.xmlsoap.org/soap/envelope/", "http://www.w3.org/*/soap-envelope", NULL},
	{"SOAP-ENC", "http://schemas.xmlsoap.org/soap/encoding/", "http://www.w3.org/*/soap-encoding", NULL},
	{"xsi", "http://www.w3.org/2001/XMLSchema-instance", "http://www.w3.org/*/XMLSchema-instance", NULL},
	{"xsd", "http://www.w3.org/2001/XMLSchema", "http://www.w3.org/*/XMLSchema", NULL},
	{"Damis", "DAMIS", NULL, NULL},
	{NULL, NULL, NULL, NULL}
};
	soap_set_namespaces(this->soap, namespaces);
};

void DAMISService::destroy()
{	soap_destroy(this->soap);
	soap_end(this->soap);
}

void DAMISService::reset()
{	destroy();
	soap_done(this->soap);
	soap_init(this->soap);
	DAMISService_init(SOAP_IO_DEFAULT, SOAP_IO_DEFAULT);
}

#ifndef WITH_PURE_VIRTUAL
DAMISService *DAMISService::copy()
{	DAMISService *dup = SOAP_NEW_COPY(DAMISService);
	if (dup)
		soap_copy_context(dup->soap, this->soap);
	return dup;
}
#endif

int DAMISService::soap_close_socket()
{	return soap_closesock(this->soap);
}

int DAMISService::soap_force_close_socket()
{	return soap_force_closesock(this->soap);
}

int DAMISService::soap_senderfault(const char *string, const char *detailXML)
{	return ::soap_sender_fault(this->soap, string, detailXML);
}

int DAMISService::soap_senderfault(const char *subcodeQName, const char *string, const char *detailXML)
{	return ::soap_sender_fault_subcode(this->soap, subcodeQName, string, detailXML);
}

int DAMISService::soap_receiverfault(const char *string, const char *detailXML)
{	return ::soap_receiver_fault(this->soap, string, detailXML);
}

int DAMISService::soap_receiverfault(const char *subcodeQName, const char *string, const char *detailXML)
{	return ::soap_receiver_fault_subcode(this->soap, subcodeQName, string, detailXML);
}

void DAMISService::soap_print_fault(FILE *fd)
{	::soap_print_fault(this->soap, fd);
}

#ifndef WITH_LEAN
#ifndef WITH_COMPAT
void DAMISService::soap_stream_fault(std::ostream& os)
{	::soap_stream_fault(this->soap, os);
}
#endif

char *DAMISService::soap_sprint_fault(char *buf, size_t len)
{	return ::soap_sprint_fault(this->soap, buf, len);
}
#endif

void DAMISService::soap_noheader()
{	this->soap->header = NULL;
}

const SOAP_ENV__Header *DAMISService::soap_header()
{	return this->soap->header;
}

int DAMISService::run(int port)
{	if (soap_valid_socket(this->soap->master) || soap_valid_socket(bind(NULL, port, 100)))
	{	for (;;)
		{	if (!soap_valid_socket(accept()) || serve())
				return this->soap->error;
			soap_destroy(this->soap);
			soap_end(this->soap);
		}
	}
	else
		return this->soap->error;
	return SOAP_OK;
}

SOAP_SOCKET DAMISService::bind(const char *host, int port, int backlog)
{	return soap_bind(this->soap, host, port, backlog);
}

SOAP_SOCKET DAMISService::accept()
{	return soap_accept(this->soap);
}

#if defined(WITH_OPENSSL) || defined(WITH_GNUTLS)
int DAMISService::ssl_accept()
{	return soap_ssl_accept(this->soap);
}
#endif

int DAMISService::serve()
{
#ifndef WITH_FASTCGI
	unsigned int k = this->soap->max_keep_alive;
#endif
	do
	{

#ifndef WITH_FASTCGI
		if (this->soap->max_keep_alive > 0 && !--k)
			this->soap->keep_alive = 0;
#endif

		if (soap_begin_serve(this->soap))
		{	if (this->soap->error >= SOAP_STOP)
				continue;
			return this->soap->error;
		}
		if (dispatch() || (this->soap->fserveloop && this->soap->fserveloop(this->soap)))
		{
#ifdef WITH_FASTCGI
			soap_send_fault(this->soap);
#else
			return soap_send_fault(this->soap);
#endif
		}

#ifdef WITH_FASTCGI
		soap_destroy(this->soap);
		soap_end(this->soap);
	} while (1);
#else
	} while (this->soap->keep_alive);
#endif
	return SOAP_OK;
}

static int serve_Damis__PCA(DAMISService*);
static int serve_Damis__SMACOFMDS(DAMISService*);
static int serve_Damis__DMA(DAMISService*);
static int serve_Damis__RELMDS(DAMISService*);
static int serve_Damis__SAMANN(DAMISService*);
static int serve_Damis__SOM(DAMISService*);
static int serve_Damis__SOMMDS(DAMISService*);
static int serve_Damis__MLP(DAMISService*);
static int serve_Damis__DF(DAMISService*);
static int serve_Damis__KMEANS(DAMISService*);
static int serve_Damis__STATPRIMITIVES(DAMISService*);
static int serve_Damis__CLEANDATA(DAMISService*);
static int serve_Damis__FILTERDATA(DAMISService*);
static int serve_Damis__SPLITDATA(DAMISService*);
static int serve_Damis__TRANSPOSEDATA(DAMISService*);
static int serve_Damis__NORMDATA(DAMISService*);

int DAMISService::dispatch()
{	DAMISService_init(this->soap->imode, this->soap->omode);
	soap_peek_element(this->soap);
	if (!soap_match_tag(this->soap, this->soap->tag, "Damis:PCA"))
		return serve_Damis__PCA(this);
	if (!soap_match_tag(this->soap, this->soap->tag, "Damis:SMACOFMDS"))
		return serve_Damis__SMACOFMDS(this);
	if (!soap_match_tag(this->soap, this->soap->tag, "Damis:DMA"))
		return serve_Damis__DMA(this);
	if (!soap_match_tag(this->soap, this->soap->tag, "Damis:RELMDS"))
		return serve_Damis__RELMDS(this);
	if (!soap_match_tag(this->soap, this->soap->tag, "Damis:SAMANN"))
		return serve_Damis__SAMANN(this);
	if (!soap_match_tag(this->soap, this->soap->tag, "Damis:SOM"))
		return serve_Damis__SOM(this);
	if (!soap_match_tag(this->soap, this->soap->tag, "Damis:SOMMDS"))
		return serve_Damis__SOMMDS(this);
	if (!soap_match_tag(this->soap, this->soap->tag, "Damis:MLP"))
		return serve_Damis__MLP(this);
	if (!soap_match_tag(this->soap, this->soap->tag, "Damis:DF"))
		return serve_Damis__DF(this);
	if (!soap_match_tag(this->soap, this->soap->tag, "Damis:KMEANS"))
		return serve_Damis__KMEANS(this);
	if (!soap_match_tag(this->soap, this->soap->tag, "Damis:STATPRIMITIVES"))
		return serve_Damis__STATPRIMITIVES(this);
	if (!soap_match_tag(this->soap, this->soap->tag, "Damis:CLEANDATA"))
		return serve_Damis__CLEANDATA(this);
	if (!soap_match_tag(this->soap, this->soap->tag, "Damis:FILTERDATA"))
		return serve_Damis__FILTERDATA(this);
	if (!soap_match_tag(this->soap, this->soap->tag, "Damis:SPLITDATA"))
		return serve_Damis__SPLITDATA(this);
	if (!soap_match_tag(this->soap, this->soap->tag, "Damis:TRANSPOSEDATA"))
		return serve_Damis__TRANSPOSEDATA(this);
	if (!soap_match_tag(this->soap, this->soap->tag, "Damis:NORMDATA"))
		return serve_Damis__NORMDATA(this);
	return this->soap->error = SOAP_NO_METHOD;
}

static int serve_Damis__PCA(DAMISService *service)
{	struct soap *soap = service->soap;
	struct Damis__PCA soap_tmp_Damis__PCA;
	struct Damis__PCAResponse _param_1;
	soap_default_Damis__PCAResponse(soap, &_param_1);
	soap_default_Damis__PCA(soap, &soap_tmp_Damis__PCA);
	soap->encodingStyle = "http://schemas.xmlsoap.org/soap/encoding/";
	if (!soap_get_Damis__PCA(soap, &soap_tmp_Damis__PCA, "Damis:PCA", NULL))
		return soap->error;
	if (soap_body_end_in(soap)
	 || soap_envelope_end_in(soap)
	 || soap_end_recv(soap))
		return soap->error;
	soap->error = service->PCA(soap_tmp_Damis__PCA.X, soap_tmp_Damis__PCA.projType, soap_tmp_Damis__PCA.d, soap_tmp_Damis__PCA.maxCalcTime, _param_1);
	if (soap->error)
		return soap->error;
	soap_serializeheader(soap);
	soap_serialize_Damis__PCAResponse(soap, &_param_1);
	if (soap_begin_count(soap))
		return soap->error;
	if (soap->mode & SOAP_IO_LENGTH)
	{	if (soap_envelope_begin_out(soap)
		 || soap_putheader(soap)
		 || soap_body_begin_out(soap)
		 || soap_put_Damis__PCAResponse(soap, &_param_1, "Damis:PCAResponse", NULL)
		 || soap_body_end_out(soap)
		 || soap_envelope_end_out(soap))
			 return soap->error;
	};
	if (soap_end_count(soap)
	 || soap_response(soap, SOAP_OK)
	 || soap_envelope_begin_out(soap)
	 || soap_putheader(soap)
	 || soap_body_begin_out(soap)
	 || soap_put_Damis__PCAResponse(soap, &_param_1, "Damis:PCAResponse", NULL)
	 || soap_body_end_out(soap)
	 || soap_envelope_end_out(soap)
	 || soap_end_send(soap))
		return soap->error;
	return soap_closesock(soap);
}

static int serve_Damis__SMACOFMDS(DAMISService *service)
{	struct soap *soap = service->soap;
	struct Damis__SMACOFMDS soap_tmp_Damis__SMACOFMDS;
	struct Damis__SMACOFMDSResponse _param_1;
	soap_default_Damis__SMACOFMDSResponse(soap, &_param_1);
	soap_default_Damis__SMACOFMDS(soap, &soap_tmp_Damis__SMACOFMDS);
	soap->encodingStyle = "http://schemas.xmlsoap.org/soap/encoding/";
	if (!soap_get_Damis__SMACOFMDS(soap, &soap_tmp_Damis__SMACOFMDS, "Damis:SMACOFMDS", NULL))
		return soap->error;
	if (soap_body_end_in(soap)
	 || soap_envelope_end_in(soap)
	 || soap_end_recv(soap))
		return soap->error;
	soap->error = service->SMACOFMDS(soap_tmp_Damis__SMACOFMDS.X, soap_tmp_Damis__SMACOFMDS.d, soap_tmp_Damis__SMACOFMDS.maxIteration, soap_tmp_Damis__SMACOFMDS.eps, soap_tmp_Damis__SMACOFMDS.zeidel, soap_tmp_Damis__SMACOFMDS.p, soap_tmp_Damis__SMACOFMDS.maxCalcTime, _param_1);
	if (soap->error)
		return soap->error;
	soap_serializeheader(soap);
	soap_serialize_Damis__SMACOFMDSResponse(soap, &_param_1);
	if (soap_begin_count(soap))
		return soap->error;
	if (soap->mode & SOAP_IO_LENGTH)
	{	if (soap_envelope_begin_out(soap)
		 || soap_putheader(soap)
		 || soap_body_begin_out(soap)
		 || soap_put_Damis__SMACOFMDSResponse(soap, &_param_1, "Damis:SMACOFMDSResponse", NULL)
		 || soap_body_end_out(soap)
		 || soap_envelope_end_out(soap))
			 return soap->error;
	};
	if (soap_end_count(soap)
	 || soap_response(soap, SOAP_OK)
	 || soap_envelope_begin_out(soap)
	 || soap_putheader(soap)
	 || soap_body_begin_out(soap)
	 || soap_put_Damis__SMACOFMDSResponse(soap, &_param_1, "Damis:SMACOFMDSResponse", NULL)
	 || soap_body_end_out(soap)
	 || soap_envelope_end_out(soap)
	 || soap_end_send(soap))
		return soap->error;
	return soap_closesock(soap);
}

static int serve_Damis__DMA(DAMISService *service)
{	struct soap *soap = service->soap;
	struct Damis__DMA soap_tmp_Damis__DMA;
	struct Damis__DMAResponse _param_1;
	soap_default_Damis__DMAResponse(soap, &_param_1);
	soap_default_Damis__DMA(soap, &soap_tmp_Damis__DMA);
	soap->encodingStyle = "http://schemas.xmlsoap.org/soap/encoding/";
	if (!soap_get_Damis__DMA(soap, &soap_tmp_Damis__DMA, "Damis:DMA", NULL))
		return soap->error;
	if (soap_body_end_in(soap)
	 || soap_envelope_end_in(soap)
	 || soap_end_recv(soap))
		return soap->error;
	soap->error = service->DMA(soap_tmp_Damis__DMA.X, soap_tmp_Damis__DMA.d, soap_tmp_Damis__DMA.maxIteration, soap_tmp_Damis__DMA.eps, soap_tmp_Damis__DMA.neighbour, soap_tmp_Damis__DMA.maxCalcTime, _param_1);
	if (soap->error)
		return soap->error;
	soap_serializeheader(soap);
	soap_serialize_Damis__DMAResponse(soap, &_param_1);
	if (soap_begin_count(soap))
		return soap->error;
	if (soap->mode & SOAP_IO_LENGTH)
	{	if (soap_envelope_begin_out(soap)
		 || soap_putheader(soap)
		 || soap_body_begin_out(soap)
		 || soap_put_Damis__DMAResponse(soap, &_param_1, "Damis:DMAResponse", NULL)
		 || soap_body_end_out(soap)
		 || soap_envelope_end_out(soap))
			 return soap->error;
	};
	if (soap_end_count(soap)
	 || soap_response(soap, SOAP_OK)
	 || soap_envelope_begin_out(soap)
	 || soap_putheader(soap)
	 || soap_body_begin_out(soap)
	 || soap_put_Damis__DMAResponse(soap, &_param_1, "Damis:DMAResponse", NULL)
	 || soap_body_end_out(soap)
	 || soap_envelope_end_out(soap)
	 || soap_end_send(soap))
		return soap->error;
	return soap_closesock(soap);
}

static int serve_Damis__RELMDS(DAMISService *service)
{	struct soap *soap = service->soap;
	struct Damis__RELMDS soap_tmp_Damis__RELMDS;
	struct Damis__RELMDSResponse _param_1;
	soap_default_Damis__RELMDSResponse(soap, &_param_1);
	soap_default_Damis__RELMDS(soap, &soap_tmp_Damis__RELMDS);
	soap->encodingStyle = "http://schemas.xmlsoap.org/soap/encoding/";
	if (!soap_get_Damis__RELMDS(soap, &soap_tmp_Damis__RELMDS, "Damis:RELMDS", NULL))
		return soap->error;
	if (soap_body_end_in(soap)
	 || soap_envelope_end_in(soap)
	 || soap_end_recv(soap))
		return soap->error;
	soap->error = service->RELMDS(soap_tmp_Damis__RELMDS.X, soap_tmp_Damis__RELMDS.d, soap_tmp_Damis__RELMDS.maxIteration, soap_tmp_Damis__RELMDS.eps, soap_tmp_Damis__RELMDS.noOfBaseVectors, soap_tmp_Damis__RELMDS.selStrategy, soap_tmp_Damis__RELMDS.maxCalcTime, _param_1);
	if (soap->error)
		return soap->error;
	soap_serializeheader(soap);
	soap_serialize_Damis__RELMDSResponse(soap, &_param_1);
	if (soap_begin_count(soap))
		return soap->error;
	if (soap->mode & SOAP_IO_LENGTH)
	{	if (soap_envelope_begin_out(soap)
		 || soap_putheader(soap)
		 || soap_body_begin_out(soap)
		 || soap_put_Damis__RELMDSResponse(soap, &_param_1, "Damis:RELMDSResponse", NULL)
		 || soap_body_end_out(soap)
		 || soap_envelope_end_out(soap))
			 return soap->error;
	};
	if (soap_end_count(soap)
	 || soap_response(soap, SOAP_OK)
	 || soap_envelope_begin_out(soap)
	 || soap_putheader(soap)
	 || soap_body_begin_out(soap)
	 || soap_put_Damis__RELMDSResponse(soap, &_param_1, "Damis:RELMDSResponse", NULL)
	 || soap_body_end_out(soap)
	 || soap_envelope_end_out(soap)
	 || soap_end_send(soap))
		return soap->error;
	return soap_closesock(soap);
}

static int serve_Damis__SAMANN(DAMISService *service)
{	struct soap *soap = service->soap;
	struct Damis__SAMANN soap_tmp_Damis__SAMANN;
	struct Damis__SAMANNResponse _param_1;
	soap_default_Damis__SAMANNResponse(soap, &_param_1);
	soap_default_Damis__SAMANN(soap, &soap_tmp_Damis__SAMANN);
	soap->encodingStyle = "http://schemas.xmlsoap.org/soap/encoding/";
	if (!soap_get_Damis__SAMANN(soap, &soap_tmp_Damis__SAMANN, "Damis:SAMANN", NULL))
		return soap->error;
	if (soap_body_end_in(soap)
	 || soap_envelope_end_in(soap)
	 || soap_end_recv(soap))
		return soap->error;
	soap->error = service->SAMANN(soap_tmp_Damis__SAMANN.X, soap_tmp_Damis__SAMANN.d, soap_tmp_Damis__SAMANN.maxIteration, soap_tmp_Damis__SAMANN.mTrain, soap_tmp_Damis__SAMANN.nNeurons, soap_tmp_Damis__SAMANN.eta, soap_tmp_Damis__SAMANN.p, soap_tmp_Damis__SAMANN.maxCalcTime, _param_1);
	if (soap->error)
		return soap->error;
	soap_serializeheader(soap);
	soap_serialize_Damis__SAMANNResponse(soap, &_param_1);
	if (soap_begin_count(soap))
		return soap->error;
	if (soap->mode & SOAP_IO_LENGTH)
	{	if (soap_envelope_begin_out(soap)
		 || soap_putheader(soap)
		 || soap_body_begin_out(soap)
		 || soap_put_Damis__SAMANNResponse(soap, &_param_1, "Damis:SAMANNResponse", NULL)
		 || soap_body_end_out(soap)
		 || soap_envelope_end_out(soap))
			 return soap->error;
	};
	if (soap_end_count(soap)
	 || soap_response(soap, SOAP_OK)
	 || soap_envelope_begin_out(soap)
	 || soap_putheader(soap)
	 || soap_body_begin_out(soap)
	 || soap_put_Damis__SAMANNResponse(soap, &_param_1, "Damis:SAMANNResponse", NULL)
	 || soap_body_end_out(soap)
	 || soap_envelope_end_out(soap)
	 || soap_end_send(soap))
		return soap->error;
	return soap_closesock(soap);
}

static int serve_Damis__SOM(DAMISService *service)
{	struct soap *soap = service->soap;
	struct Damis__SOM soap_tmp_Damis__SOM;
	struct Damis__SOMResponse _param_1;
	soap_default_Damis__SOMResponse(soap, &_param_1);
	soap_default_Damis__SOM(soap, &soap_tmp_Damis__SOM);
	soap->encodingStyle = "http://schemas.xmlsoap.org/soap/encoding/";
	if (!soap_get_Damis__SOM(soap, &soap_tmp_Damis__SOM, "Damis:SOM", NULL))
		return soap->error;
	if (soap_body_end_in(soap)
	 || soap_envelope_end_in(soap)
	 || soap_end_recv(soap))
		return soap->error;
	soap->error = service->SOM(soap_tmp_Damis__SOM.X, soap_tmp_Damis__SOM.rows, soap_tmp_Damis__SOM.columns, soap_tmp_Damis__SOM.eHat, soap_tmp_Damis__SOM.p, soap_tmp_Damis__SOM.maxCalcTime, _param_1);
	if (soap->error)
		return soap->error;
	soap_serializeheader(soap);
	soap_serialize_Damis__SOMResponse(soap, &_param_1);
	if (soap_begin_count(soap))
		return soap->error;
	if (soap->mode & SOAP_IO_LENGTH)
	{	if (soap_envelope_begin_out(soap)
		 || soap_putheader(soap)
		 || soap_body_begin_out(soap)
		 || soap_put_Damis__SOMResponse(soap, &_param_1, "Damis:SOMResponse", NULL)
		 || soap_body_end_out(soap)
		 || soap_envelope_end_out(soap))
			 return soap->error;
	};
	if (soap_end_count(soap)
	 || soap_response(soap, SOAP_OK)
	 || soap_envelope_begin_out(soap)
	 || soap_putheader(soap)
	 || soap_body_begin_out(soap)
	 || soap_put_Damis__SOMResponse(soap, &_param_1, "Damis:SOMResponse", NULL)
	 || soap_body_end_out(soap)
	 || soap_envelope_end_out(soap)
	 || soap_end_send(soap))
		return soap->error;
	return soap_closesock(soap);
}

static int serve_Damis__SOMMDS(DAMISService *service)
{	struct soap *soap = service->soap;
	struct Damis__SOMMDS soap_tmp_Damis__SOMMDS;
	struct Damis__SOMMDSResponse _param_1;
	soap_default_Damis__SOMMDSResponse(soap, &_param_1);
	soap_default_Damis__SOMMDS(soap, &soap_tmp_Damis__SOMMDS);
	soap->encodingStyle = "http://schemas.xmlsoap.org/soap/encoding/";
	if (!soap_get_Damis__SOMMDS(soap, &soap_tmp_Damis__SOMMDS, "Damis:SOMMDS", NULL))
		return soap->error;
	if (soap_body_end_in(soap)
	 || soap_envelope_end_in(soap)
	 || soap_end_recv(soap))
		return soap->error;
	soap->error = service->SOMMDS(soap_tmp_Damis__SOMMDS.X, soap_tmp_Damis__SOMMDS.rows, soap_tmp_Damis__SOMMDS.columns, soap_tmp_Damis__SOMMDS.eHat, soap_tmp_Damis__SOMMDS.mdsIteration, soap_tmp_Damis__SOMMDS.eps, soap_tmp_Damis__SOMMDS.mdsProjection, soap_tmp_Damis__SOMMDS.maxCalcTime, _param_1);
	if (soap->error)
		return soap->error;
	soap_serializeheader(soap);
	soap_serialize_Damis__SOMMDSResponse(soap, &_param_1);
	if (soap_begin_count(soap))
		return soap->error;
	if (soap->mode & SOAP_IO_LENGTH)
	{	if (soap_envelope_begin_out(soap)
		 || soap_putheader(soap)
		 || soap_body_begin_out(soap)
		 || soap_put_Damis__SOMMDSResponse(soap, &_param_1, "Damis:SOMMDSResponse", NULL)
		 || soap_body_end_out(soap)
		 || soap_envelope_end_out(soap))
			 return soap->error;
	};
	if (soap_end_count(soap)
	 || soap_response(soap, SOAP_OK)
	 || soap_envelope_begin_out(soap)
	 || soap_putheader(soap)
	 || soap_body_begin_out(soap)
	 || soap_put_Damis__SOMMDSResponse(soap, &_param_1, "Damis:SOMMDSResponse", NULL)
	 || soap_body_end_out(soap)
	 || soap_envelope_end_out(soap)
	 || soap_end_send(soap))
		return soap->error;
	return soap_closesock(soap);
}

static int serve_Damis__MLP(DAMISService *service)
{	struct soap *soap = service->soap;
	struct Damis__MLP soap_tmp_Damis__MLP;
	struct Damis__MLPResponse _param_1;
	soap_default_Damis__MLPResponse(soap, &_param_1);
	soap_default_Damis__MLP(soap, &soap_tmp_Damis__MLP);
	soap->encodingStyle = "http://schemas.xmlsoap.org/soap/encoding/";
	if (!soap_get_Damis__MLP(soap, &soap_tmp_Damis__MLP, "Damis:MLP", NULL))
		return soap->error;
	if (soap_body_end_in(soap)
	 || soap_envelope_end_in(soap)
	 || soap_end_recv(soap))
		return soap->error;
	soap->error = service->MLP(soap_tmp_Damis__MLP.X, soap_tmp_Damis__MLP.h1pNo, soap_tmp_Damis__MLP.h2pNo, soap_tmp_Damis__MLP.qty, soap_tmp_Damis__MLP.kFoldValidation, soap_tmp_Damis__MLP.maxIteration, soap_tmp_Damis__MLP.p, soap_tmp_Damis__MLP.maxCalcTime, _param_1);
	if (soap->error)
		return soap->error;
	soap_serializeheader(soap);
	soap_serialize_Damis__MLPResponse(soap, &_param_1);
	if (soap_begin_count(soap))
		return soap->error;
	if (soap->mode & SOAP_IO_LENGTH)
	{	if (soap_envelope_begin_out(soap)
		 || soap_putheader(soap)
		 || soap_body_begin_out(soap)
		 || soap_put_Damis__MLPResponse(soap, &_param_1, "Damis:MLPResponse", NULL)
		 || soap_body_end_out(soap)
		 || soap_envelope_end_out(soap))
			 return soap->error;
	};
	if (soap_end_count(soap)
	 || soap_response(soap, SOAP_OK)
	 || soap_envelope_begin_out(soap)
	 || soap_putheader(soap)
	 || soap_body_begin_out(soap)
	 || soap_put_Damis__MLPResponse(soap, &_param_1, "Damis:MLPResponse", NULL)
	 || soap_body_end_out(soap)
	 || soap_envelope_end_out(soap)
	 || soap_end_send(soap))
		return soap->error;
	return soap_closesock(soap);
}

static int serve_Damis__DF(DAMISService *service)
{	struct soap *soap = service->soap;
	struct Damis__DF soap_tmp_Damis__DF;
	struct Damis__DFResponse _param_1;
	soap_default_Damis__DFResponse(soap, &_param_1);
	soap_default_Damis__DF(soap, &soap_tmp_Damis__DF);
	soap->encodingStyle = "http://schemas.xmlsoap.org/soap/encoding/";
	if (!soap_get_Damis__DF(soap, &soap_tmp_Damis__DF, "Damis:DF", NULL))
		return soap->error;
	if (soap_body_end_in(soap)
	 || soap_envelope_end_in(soap)
	 || soap_end_recv(soap))
		return soap->error;
	soap->error = service->DF(soap_tmp_Damis__DF.X, soap_tmp_Damis__DF.r, soap_tmp_Damis__DF.dL, soap_tmp_Damis__DF.dT, soap_tmp_Damis__DF.maxCalcTime, _param_1);
	if (soap->error)
		return soap->error;
	soap_serializeheader(soap);
	soap_serialize_Damis__DFResponse(soap, &_param_1);
	if (soap_begin_count(soap))
		return soap->error;
	if (soap->mode & SOAP_IO_LENGTH)
	{	if (soap_envelope_begin_out(soap)
		 || soap_putheader(soap)
		 || soap_body_begin_out(soap)
		 || soap_put_Damis__DFResponse(soap, &_param_1, "Damis:DFResponse", NULL)
		 || soap_body_end_out(soap)
		 || soap_envelope_end_out(soap))
			 return soap->error;
	};
	if (soap_end_count(soap)
	 || soap_response(soap, SOAP_OK)
	 || soap_envelope_begin_out(soap)
	 || soap_putheader(soap)
	 || soap_body_begin_out(soap)
	 || soap_put_Damis__DFResponse(soap, &_param_1, "Damis:DFResponse", NULL)
	 || soap_body_end_out(soap)
	 || soap_envelope_end_out(soap)
	 || soap_end_send(soap))
		return soap->error;
	return soap_closesock(soap);
}

static int serve_Damis__KMEANS(DAMISService *service)
{	struct soap *soap = service->soap;
	struct Damis__KMEANS soap_tmp_Damis__KMEANS;
	struct Damis__KMEANSResponse _param_1;
	soap_default_Damis__KMEANSResponse(soap, &_param_1);
	soap_default_Damis__KMEANS(soap, &soap_tmp_Damis__KMEANS);
	soap->encodingStyle = "http://schemas.xmlsoap.org/soap/encoding/";
	if (!soap_get_Damis__KMEANS(soap, &soap_tmp_Damis__KMEANS, "Damis:KMEANS", NULL))
		return soap->error;
	if (soap_body_end_in(soap)
	 || soap_envelope_end_in(soap)
	 || soap_end_recv(soap))
		return soap->error;
	soap->error = service->KMEANS(soap_tmp_Damis__KMEANS.X, soap_tmp_Damis__KMEANS.kMax, soap_tmp_Damis__KMEANS.maxIteration, soap_tmp_Damis__KMEANS.maxCalcTime, _param_1);
	if (soap->error)
		return soap->error;
	soap_serializeheader(soap);
	soap_serialize_Damis__KMEANSResponse(soap, &_param_1);
	if (soap_begin_count(soap))
		return soap->error;
	if (soap->mode & SOAP_IO_LENGTH)
	{	if (soap_envelope_begin_out(soap)
		 || soap_putheader(soap)
		 || soap_body_begin_out(soap)
		 || soap_put_Damis__KMEANSResponse(soap, &_param_1, "Damis:KMEANSResponse", NULL)
		 || soap_body_end_out(soap)
		 || soap_envelope_end_out(soap))
			 return soap->error;
	};
	if (soap_end_count(soap)
	 || soap_response(soap, SOAP_OK)
	 || soap_envelope_begin_out(soap)
	 || soap_putheader(soap)
	 || soap_body_begin_out(soap)
	 || soap_put_Damis__KMEANSResponse(soap, &_param_1, "Damis:KMEANSResponse", NULL)
	 || soap_body_end_out(soap)
	 || soap_envelope_end_out(soap)
	 || soap_end_send(soap))
		return soap->error;
	return soap_closesock(soap);
}

static int serve_Damis__STATPRIMITIVES(DAMISService *service)
{	struct soap *soap = service->soap;
	struct Damis__STATPRIMITIVES soap_tmp_Damis__STATPRIMITIVES;
	struct Damis__STATPRIMITIVESResponse _param_1;
	soap_default_Damis__STATPRIMITIVESResponse(soap, &_param_1);
	soap_default_Damis__STATPRIMITIVES(soap, &soap_tmp_Damis__STATPRIMITIVES);
	soap->encodingStyle = "http://schemas.xmlsoap.org/soap/encoding/";
	if (!soap_get_Damis__STATPRIMITIVES(soap, &soap_tmp_Damis__STATPRIMITIVES, "Damis:STATPRIMITIVES", NULL))
		return soap->error;
	if (soap_body_end_in(soap)
	 || soap_envelope_end_in(soap)
	 || soap_end_recv(soap))
		return soap->error;
	soap->error = service->STATPRIMITIVES(soap_tmp_Damis__STATPRIMITIVES.X, soap_tmp_Damis__STATPRIMITIVES.maxCalcTime, _param_1);
	if (soap->error)
		return soap->error;
	soap_serializeheader(soap);
	soap_serialize_Damis__STATPRIMITIVESResponse(soap, &_param_1);
	if (soap_begin_count(soap))
		return soap->error;
	if (soap->mode & SOAP_IO_LENGTH)
	{	if (soap_envelope_begin_out(soap)
		 || soap_putheader(soap)
		 || soap_body_begin_out(soap)
		 || soap_put_Damis__STATPRIMITIVESResponse(soap, &_param_1, "Damis:STATPRIMITIVESResponse", NULL)
		 || soap_body_end_out(soap)
		 || soap_envelope_end_out(soap))
			 return soap->error;
	};
	if (soap_end_count(soap)
	 || soap_response(soap, SOAP_OK)
	 || soap_envelope_begin_out(soap)
	 || soap_putheader(soap)
	 || soap_body_begin_out(soap)
	 || soap_put_Damis__STATPRIMITIVESResponse(soap, &_param_1, "Damis:STATPRIMITIVESResponse", NULL)
	 || soap_body_end_out(soap)
	 || soap_envelope_end_out(soap)
	 || soap_end_send(soap))
		return soap->error;
	return soap_closesock(soap);
}

static int serve_Damis__CLEANDATA(DAMISService *service)
{	struct soap *soap = service->soap;
	struct Damis__CLEANDATA soap_tmp_Damis__CLEANDATA;
	struct Damis__CLEANDATAResponse _param_1;
	soap_default_Damis__CLEANDATAResponse(soap, &_param_1);
	soap_default_Damis__CLEANDATA(soap, &soap_tmp_Damis__CLEANDATA);
	soap->encodingStyle = "http://schemas.xmlsoap.org/soap/encoding/";
	if (!soap_get_Damis__CLEANDATA(soap, &soap_tmp_Damis__CLEANDATA, "Damis:CLEANDATA", NULL))
		return soap->error;
	if (soap_body_end_in(soap)
	 || soap_envelope_end_in(soap)
	 || soap_end_recv(soap))
		return soap->error;
	soap->error = service->CLEANDATA(soap_tmp_Damis__CLEANDATA.X, soap_tmp_Damis__CLEANDATA.maxCalcTime, _param_1);
	if (soap->error)
		return soap->error;
	soap_serializeheader(soap);
	soap_serialize_Damis__CLEANDATAResponse(soap, &_param_1);
	if (soap_begin_count(soap))
		return soap->error;
	if (soap->mode & SOAP_IO_LENGTH)
	{	if (soap_envelope_begin_out(soap)
		 || soap_putheader(soap)
		 || soap_body_begin_out(soap)
		 || soap_put_Damis__CLEANDATAResponse(soap, &_param_1, "Damis:CLEANDATAResponse", NULL)
		 || soap_body_end_out(soap)
		 || soap_envelope_end_out(soap))
			 return soap->error;
	};
	if (soap_end_count(soap)
	 || soap_response(soap, SOAP_OK)
	 || soap_envelope_begin_out(soap)
	 || soap_putheader(soap)
	 || soap_body_begin_out(soap)
	 || soap_put_Damis__CLEANDATAResponse(soap, &_param_1, "Damis:CLEANDATAResponse", NULL)
	 || soap_body_end_out(soap)
	 || soap_envelope_end_out(soap)
	 || soap_end_send(soap))
		return soap->error;
	return soap_closesock(soap);
}

static int serve_Damis__FILTERDATA(DAMISService *service)
{	struct soap *soap = service->soap;
	struct Damis__FILTERDATA soap_tmp_Damis__FILTERDATA;
	struct Damis__FILTERDATAResponse _param_1;
	soap_default_Damis__FILTERDATAResponse(soap, &_param_1);
	soap_default_Damis__FILTERDATA(soap, &soap_tmp_Damis__FILTERDATA);
	soap->encodingStyle = "http://schemas.xmlsoap.org/soap/encoding/";
	if (!soap_get_Damis__FILTERDATA(soap, &soap_tmp_Damis__FILTERDATA, "Damis:FILTERDATA", NULL))
		return soap->error;
	if (soap_body_end_in(soap)
	 || soap_envelope_end_in(soap)
	 || soap_end_recv(soap))
		return soap->error;
	soap->error = service->FILTERDATA(soap_tmp_Damis__FILTERDATA.X, soap_tmp_Damis__FILTERDATA.retFilteredData, soap_tmp_Damis__FILTERDATA.zValue, soap_tmp_Damis__FILTERDATA.attrIndex, soap_tmp_Damis__FILTERDATA.maxCalcTime, _param_1);
	if (soap->error)
		return soap->error;
	soap_serializeheader(soap);
	soap_serialize_Damis__FILTERDATAResponse(soap, &_param_1);
	if (soap_begin_count(soap))
		return soap->error;
	if (soap->mode & SOAP_IO_LENGTH)
	{	if (soap_envelope_begin_out(soap)
		 || soap_putheader(soap)
		 || soap_body_begin_out(soap)
		 || soap_put_Damis__FILTERDATAResponse(soap, &_param_1, "Damis:FILTERDATAResponse", NULL)
		 || soap_body_end_out(soap)
		 || soap_envelope_end_out(soap))
			 return soap->error;
	};
	if (soap_end_count(soap)
	 || soap_response(soap, SOAP_OK)
	 || soap_envelope_begin_out(soap)
	 || soap_putheader(soap)
	 || soap_body_begin_out(soap)
	 || soap_put_Damis__FILTERDATAResponse(soap, &_param_1, "Damis:FILTERDATAResponse", NULL)
	 || soap_body_end_out(soap)
	 || soap_envelope_end_out(soap)
	 || soap_end_send(soap))
		return soap->error;
	return soap_closesock(soap);
}

static int serve_Damis__SPLITDATA(DAMISService *service)
{	struct soap *soap = service->soap;
	struct Damis__SPLITDATA soap_tmp_Damis__SPLITDATA;
	struct Damis__SPLITDATAResponse _param_1;
	soap_default_Damis__SPLITDATAResponse(soap, &_param_1);
	soap_default_Damis__SPLITDATA(soap, &soap_tmp_Damis__SPLITDATA);
	soap->encodingStyle = "http://schemas.xmlsoap.org/soap/encoding/";
	if (!soap_get_Damis__SPLITDATA(soap, &soap_tmp_Damis__SPLITDATA, "Damis:SPLITDATA", NULL))
		return soap->error;
	if (soap_body_end_in(soap)
	 || soap_envelope_end_in(soap)
	 || soap_end_recv(soap))
		return soap->error;
	soap->error = service->SPLITDATA(soap_tmp_Damis__SPLITDATA.X, soap_tmp_Damis__SPLITDATA.reshufleObjects, soap_tmp_Damis__SPLITDATA.firstSubsetPerc, soap_tmp_Damis__SPLITDATA.secondSubsetPerc, soap_tmp_Damis__SPLITDATA.maxCalcTime, _param_1);
	if (soap->error)
		return soap->error;
	soap_serializeheader(soap);
	soap_serialize_Damis__SPLITDATAResponse(soap, &_param_1);
	if (soap_begin_count(soap))
		return soap->error;
	if (soap->mode & SOAP_IO_LENGTH)
	{	if (soap_envelope_begin_out(soap)
		 || soap_putheader(soap)
		 || soap_body_begin_out(soap)
		 || soap_put_Damis__SPLITDATAResponse(soap, &_param_1, "Damis:SPLITDATAResponse", NULL)
		 || soap_body_end_out(soap)
		 || soap_envelope_end_out(soap))
			 return soap->error;
	};
	if (soap_end_count(soap)
	 || soap_response(soap, SOAP_OK)
	 || soap_envelope_begin_out(soap)
	 || soap_putheader(soap)
	 || soap_body_begin_out(soap)
	 || soap_put_Damis__SPLITDATAResponse(soap, &_param_1, "Damis:SPLITDATAResponse", NULL)
	 || soap_body_end_out(soap)
	 || soap_envelope_end_out(soap)
	 || soap_end_send(soap))
		return soap->error;
	return soap_closesock(soap);
}

static int serve_Damis__TRANSPOSEDATA(DAMISService *service)
{	struct soap *soap = service->soap;
	struct Damis__TRANSPOSEDATA soap_tmp_Damis__TRANSPOSEDATA;
	struct Damis__TRANSPOSEDATAResponse _param_1;
	soap_default_Damis__TRANSPOSEDATAResponse(soap, &_param_1);
	soap_default_Damis__TRANSPOSEDATA(soap, &soap_tmp_Damis__TRANSPOSEDATA);
	soap->encodingStyle = "http://schemas.xmlsoap.org/soap/encoding/";
	if (!soap_get_Damis__TRANSPOSEDATA(soap, &soap_tmp_Damis__TRANSPOSEDATA, "Damis:TRANSPOSEDATA", NULL))
		return soap->error;
	if (soap_body_end_in(soap)
	 || soap_envelope_end_in(soap)
	 || soap_end_recv(soap))
		return soap->error;
	soap->error = service->TRANSPOSEDATA(soap_tmp_Damis__TRANSPOSEDATA.X, soap_tmp_Damis__TRANSPOSEDATA.maxCalcTime, _param_1);
	if (soap->error)
		return soap->error;
	soap_serializeheader(soap);
	soap_serialize_Damis__TRANSPOSEDATAResponse(soap, &_param_1);
	if (soap_begin_count(soap))
		return soap->error;
	if (soap->mode & SOAP_IO_LENGTH)
	{	if (soap_envelope_begin_out(soap)
		 || soap_putheader(soap)
		 || soap_body_begin_out(soap)
		 || soap_put_Damis__TRANSPOSEDATAResponse(soap, &_param_1, "Damis:TRANSPOSEDATAResponse", NULL)
		 || soap_body_end_out(soap)
		 || soap_envelope_end_out(soap))
			 return soap->error;
	};
	if (soap_end_count(soap)
	 || soap_response(soap, SOAP_OK)
	 || soap_envelope_begin_out(soap)
	 || soap_putheader(soap)
	 || soap_body_begin_out(soap)
	 || soap_put_Damis__TRANSPOSEDATAResponse(soap, &_param_1, "Damis:TRANSPOSEDATAResponse", NULL)
	 || soap_body_end_out(soap)
	 || soap_envelope_end_out(soap)
	 || soap_end_send(soap))
		return soap->error;
	return soap_closesock(soap);
}

static int serve_Damis__NORMDATA(DAMISService *service)
{	struct soap *soap = service->soap;
	struct Damis__NORMDATA soap_tmp_Damis__NORMDATA;
	struct Damis__NORMDATAResponse _param_1;
	soap_default_Damis__NORMDATAResponse(soap, &_param_1);
	soap_default_Damis__NORMDATA(soap, &soap_tmp_Damis__NORMDATA);
	soap->encodingStyle = "http://schemas.xmlsoap.org/soap/encoding/";
	if (!soap_get_Damis__NORMDATA(soap, &soap_tmp_Damis__NORMDATA, "Damis:NORMDATA", NULL))
		return soap->error;
	if (soap_body_end_in(soap)
	 || soap_envelope_end_in(soap)
	 || soap_end_recv(soap))
		return soap->error;
	soap->error = service->NORMDATA(soap_tmp_Damis__NORMDATA.X, soap_tmp_Damis__NORMDATA.normMeanStd, soap_tmp_Damis__NORMDATA.a, soap_tmp_Damis__NORMDATA.b, soap_tmp_Damis__NORMDATA.maxCalcTime, _param_1);
	if (soap->error)
		return soap->error;
	soap_serializeheader(soap);
	soap_serialize_Damis__NORMDATAResponse(soap, &_param_1);
	if (soap_begin_count(soap))
		return soap->error;
	if (soap->mode & SOAP_IO_LENGTH)
	{	if (soap_envelope_begin_out(soap)
		 || soap_putheader(soap)
		 || soap_body_begin_out(soap)
		 || soap_put_Damis__NORMDATAResponse(soap, &_param_1, "Damis:NORMDATAResponse", NULL)
		 || soap_body_end_out(soap)
		 || soap_envelope_end_out(soap))
			 return soap->error;
	};
	if (soap_end_count(soap)
	 || soap_response(soap, SOAP_OK)
	 || soap_envelope_begin_out(soap)
	 || soap_putheader(soap)
	 || soap_body_begin_out(soap)
	 || soap_put_Damis__NORMDATAResponse(soap, &_param_1, "Damis:NORMDATAResponse", NULL)
	 || soap_body_end_out(soap)
	 || soap_envelope_end_out(soap)
	 || soap_end_send(soap))
		return soap->error;
	return soap_closesock(soap);
}
/* End of server object code */
